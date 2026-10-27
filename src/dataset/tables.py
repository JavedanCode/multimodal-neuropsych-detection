"""Chunked participant aggregation and split-preserving Stage 3 outputs."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import logging
from pathlib import Path
import tarfile
import numpy as np
import pandas as pd
import yaml
from src.dataset.inventory import load_split_labels, check_label_consistency, _is_unsafe_member
from src.dataset.preprocessing import ParticipantPreprocessor

LOG = logging.getLogger(__name__)
VERSION = 1


class ForwardReader(io.RawIOBase):
    def __init__(self, stream):
        self.stream = stream
    def readable(self):
        return True
    def readinto(self, buffer):
        data = self.stream.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def aggregate(stream, columns: list[str], modality: str, chunk_rows=10000, confidence_threshold=0.8):
    """Merge chunk moments using population variance; no whole recording in RAM."""
    count = np.zeros(len(columns), dtype=np.int64)
    mean = np.zeros(len(columns))
    m2 = np.zeros(len(columns))
    total = accepted = invalid = confidence_n = 0
    confidence_sum = 0.0
    delimiter = ';' if modality == 'audio' else ','
    reader = pd.read_csv(io.BufferedReader(ForwardReader(stream)), sep=delimiter, chunksize=chunk_rows)
    for chunk in reader:
        chunk.columns = chunk.columns.str.strip()
        if not set(columns).issubset(chunk.columns):
            raise ValueError('Feature schema changed within input file')
        total += len(chunk)
        values = chunk[columns].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        finite = np.isfinite(values)
        invalid += int((~finite).sum())
        if modality == 'facial':
            confidence = pd.to_numeric(chunk['confidence'], errors='coerce').to_numpy(float)
            success = pd.to_numeric(chunk['success'], errors='coerce').to_numpy(float)
            valid_confidence = np.isfinite(confidence) & (confidence >= 0) & (confidence <= 1)
            confidence_sum += float(confidence[valid_confidence].sum())
            confidence_n += int(valid_confidence.sum())
            keep = valid_confidence & (confidence >= confidence_threshold) & (success == 1)
            values, finite = values[keep], finite[keep]
        accepted += len(values)
        n = finite.sum(axis=0)
        sums = np.where(finite, values, 0).sum(axis=0)
        batch_mean = np.divide(sums, n, out=np.zeros_like(mean), where=n > 0)
        batch_m2 = np.where(finite, values - batch_mean, 0) ** 2
        combined = count + n
        delta = batch_mean - mean
        m2 += batch_m2.sum(axis=0) + np.divide(delta**2 * count * n, combined, out=np.zeros_like(mean), where=combined > 0)
        mean += np.divide(delta * n, combined, out=np.zeros_like(mean), where=combined > 0)
        count = combined
    std = np.sqrt(np.maximum(np.divide(m2, count, out=np.zeros_like(mean), where=count > 0), 0))
    features = {}
    for i, column in enumerate(columns):
        features[f'{modality}__{column}__mean'] = float(mean[i]) if count[i] else None
        features[f'{modality}__{column}__std'] = float(std[i]) if count[i] else None
    quality = {'available': float(bool(count.any())), 'invalid_fraction': invalid/(total*len(columns)) if total else 1.0,
               'retained_fraction': accepted/total if total else 0.0}
    if modality == 'facial':
        quality['confidence_mean'] = confidence_sum/confidence_n if confidence_n else 0.0
        quality['confidence_valid_fraction'] = confidence_n/total if total else 0.0
    return features, quality, {'rows': total, 'retained_rows': accepted, 'finite_counts': dict(zip(columns, map(int, count)))}


def participant(path: Path, schemas: dict, settings: dict) -> dict:
    result = {'features': {}, 'quality': {}, 'diagnostics': {}}
    found = set()
    with tarfile.open(path, 'r|gz') as archive:
        for member in archive:
            if _is_unsafe_member(member.name) or member.issym() or member.islnk():
                raise ValueError(f'Unsafe archive member in {path.name}')
            for modality, schema in schemas.items():
                expected = f'{int(path.name.split("_")[0])}_{schema["suffix"]}'
                if Path(member.name).name != expected:
                    continue
                if modality in found or not member.isfile():
                    raise ValueError(f'Duplicate/nonregular feature file in {path.name}')
                stream = archive.extractfile(member)
                f, q, d = aggregate(stream, schema['columns'], modality, **settings)
                result['features'].update(f)
                result['quality'].update({f'{modality}__{key}': value for key,value in q.items()})
                result['diagnostics'][modality] = d
                found.add(modality)
    if found != set(schemas):
        raise ValueError(f'Missing required feature file in {path.name}')
    return result


def build(root: Path, inventory_path: Path, output: Path, settings: dict, workers=2):
    root, output = root.resolve(), output.resolve()
    if output.is_relative_to(root):
        raise ValueError('Outputs must be outside raw data')
    if workers < 1 or workers > 4 or settings['chunk_rows'] < 1 or not 0 <= settings['confidence_threshold'] <= 1:
        raise ValueError('Invalid worker/chunk/threshold configuration')
    inventory = json.loads(inventory_path.read_text(encoding='utf-8'))
    if Path(inventory['root']).resolve() != root:
        raise ValueError('Feature inventory belongs to another dataset root; rerun Stage 2')
    frames = {}
    for split in ('train','dev','test'):
        frame, info = load_split_labels(root/f'{split}_split.csv')
        if info.duplicate_ids:
            raise ValueError('Duplicate participants in split')
        frames[split] = frame[['Participant_ID','PHQ_Binary']].assign(split=split)
    if check_label_consistency(frames)['ids_in_multiple_splits']:
        raise ValueError('Participant overlap across splits')
    labels = pd.concat(frames.values()).set_index('Participant_ID').sort_index()
    if labels.empty:
        raise ValueError('No labels')
    schemas = {}
    for modality, suffix, excluded in [('audio','OpenSMILE2.3.0_egemaps.csv',{'name','frameTime'}),('facial','OpenFace2.1.0_Pose_gaze_AUs.csv',{'frame','timestamp','confidence','success'})]:
        matching = [f for a in inventory['archives'].values() for f in a['files'] if Path(f['name']).name.endswith('_'+suffix)]
        variants = {tuple(f['columns']) for f in matching}
        if len(variants) != 1 or len(matching) != len(labels):
            raise ValueError(f'Incomplete/inconsistent {modality} schema inventory')
        columns = [c for c in next(iter(variants)) if c not in excluded]
        probe = pd.DataFrame(columns=[f'{modality}__{c}' for c in columns])
        ParticipantPreprocessor._validate(probe)
        schemas[modality] = {'suffix':suffix,'columns':columns}
    output.mkdir(parents=True,exist_ok=True)
    cache = output/'participant_cache'; cache.mkdir(exist_ok=True)
    contract = json.dumps({'version':VERSION,'root':str(root),'settings':settings,'schemas':schemas},sort_keys=True)
    fingerprint = hashlib.sha256(contract.encode()).hexdigest()
    def load_one(pid):
        path=root/f'{pid}_P.tar.gz'
        stat=path.stat()
        prior=inventory['archives'].get(path.name,{})
        from datetime import datetime,timezone
        if prior.get('size_bytes') != stat.st_size or prior.get('mtime_utc') != datetime.fromtimestamp(stat.st_mtime,tz=timezone.utc).isoformat():
            raise ValueError(f'Stale feature inventory for {path.name}')
        key={'contract':fingerprint,'size':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        target=cache/f'{pid}.json'
        if target.exists():
            saved=json.loads(target.read_text())
            if saved.get('key')==key:
                return pid,saved['data']
        data=participant(path,schemas,settings)
        temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps({'key':key,'data':data},allow_nan=False),encoding='utf-8')
        temporary.replace(target)
        return pid,data
    rows={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i,(pid,row) in enumerate(pool.map(load_one, labels.index),1):
            rows[pid]=row
            LOG.info('Aggregated %d/%d participants',i,len(labels))
    features=pd.DataFrame.from_dict({p:r['features'] for p,r in rows.items()},orient='index').sort_index().astype(float)
    quality=pd.DataFrame.from_dict({p:r['quality'] for p,r in rows.items()},orient='index').sort_index()
    for table in (features,quality):
        table.index.name='Participant_ID'
    train=labels.index[labels['split']=='train']
    preprocess=ParticipantPreprocessor().fit(features.loc[train])
    normalized=preprocess.transform(features)
    features.to_parquet(output/'features.parquet')
    quality.to_parquet(output/'quality.parquet')
    labels.to_parquet(output/'labels.parquet')
    for split in frames:
        normalized.loc[labels.index[labels['split']==split]].to_parquet(output/f'{split}_normalized.parquet')
    state={'fit_split':'train','fit_participant_ids':list(map(int,train)), 'input_columns':list(features), 'output_columns':preprocess.columns_, 'dropped':preprocess.dropped_, 'medians':preprocess.medians_.to_dict(), 'means':preprocess.means_.to_dict(), 'scales':preprocess.scales_.to_dict()}
    (output/'preprocessing.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    report={'participants':len(labels),'split_counts':labels['split'].value_counts().to_dict(),'raw_features':len(features.columns),'retained_features':len(normalized.columns),'settings':settings,'schemas':schemas,'missing_cells_by_split':{s:int(features.loc[labels.index[labels['split']==s]].isna().sum().sum()) for s in frames},'quality_ranges':{c:{'min':float(quality[c].min()),'max':float(quality[c].max())} for c in quality},'dropped_train_features':preprocess.dropped_,'exclusions':[], 'limitations':['Session-level aggregation; no temporal audio/facial alignment or speaker diarization.', 'Confidence cutoff is a fixed engineering choice, not a validated reliability score.', 'Label semantics remain unresolved; no classifier training performed.', 'For CV/HHO use raw features.parquet and fit preprocessing inside each training fold, never these pre-normalized tables.']}
    (output/'stage3_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/default.yaml'))
    args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    project=Path(__file__).resolve().parents[2]
    paths=config['paths']; settings=config.get('aggregation',{})
    resolve=lambda p: project/Path(p)
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    report=build(resolve(paths['data_root']),resolve(paths.get('feature_inventory','data/manifests/feature_inventory.json')),resolve(paths['processed']),{'chunk_rows':settings.get('chunk_rows',10000),'confidence_threshold':settings.get('confidence_threshold',0.8)},workers=settings.get('workers',2))
    LOG.info('Complete: %d participants, %d retained features', report['participants'],report['retained_features'])


if __name__=='__main__':
    main()
