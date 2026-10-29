"""Training-only transcript-covered summaries with explicit timing quarantine."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import logging
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

from src.dataset.audio_summary import MAX_BYTES, MAX_ROWS
from src.dataset.inventory import _is_unsafe_member
from src.dataset.label_speech_audit import transcript_intervals, covered_frames
from src.experiments.baselines import sha256


def moments(frame, columns, intervals):
    times=pd.to_numeric(frame.frameTime, errors='coerce').to_numpy(float)
    if not len(times) or len(times)>MAX_ROWS or not np.isfinite(times).all() or (times<0).any() or (np.diff(times)<0).any():
        raise ValueError('Invalid feature timeline')
    mask=covered_frames(times,intervals)
    if not mask.any():raise ValueError('No transcript-covered frames')
    result={}
    for c in columns:
        v=pd.to_numeric(frame.loc[mask,c],errors='coerce').to_numpy(float)
        v=v[np.isfinite(v)]
        result[f'audio__{c}__mean']=float(v.mean()) if len(v) else None
        result[f'audio__{c}__std']=float(v.std()) if len(v) else None
    return result,float(mask.mean())


def extract(path, columns, duration):
    pid=int(path.name.split('_')[0]); found={};hashes={}
    wanted={f'{pid}_Transcript.csv':16*1024*1024,
            f'{pid}_OpenSMILE2.3.0_egemaps.csv':MAX_BYTES}
    with tarfile.open(path,'r|gz') as archive:
        for member in archive:
            if _is_unsafe_member(member.name) or member.issym() or member.islnk():raise ValueError('Unsafe member')
            name=Path(member.name).name
            if name not in wanted:continue
            if name in found or not member.isfile() or member.size>wanted[name]:raise ValueError('Invalid member')
            raw=archive.extractfile(member).read(wanted[name]+1)
            if len(raw)!=member.size:raise ValueError('Member size mismatch')
            hashes[name]=hashlib.sha256(raw).hexdigest()
            found[name]=pd.read_csv(io.BytesIO(raw),sep=',' if 'Transcript' in name else ';',nrows=MAX_ROWS+1)
            if len(found)==2:break
    if len(found)!=2:raise ValueError('Missing transcript/features')
    intervals,stats=transcript_intervals(found[f'{pid}_Transcript.csv'],duration)
    if stats['invalid_or_empty_intervals']:raise ValueError('Invalid training intervals')
    features,coverage=moments(found[f'{pid}_OpenSMILE2.3.0_egemaps.csv'],columns,intervals)
    return {'features':features,'coverage':coverage,'member_hashes':hashes}


def build(project):
    processed=project/'data/processed';labels=pd.read_parquet(processed/'labels.parquet')
    ids=labels.index[labels.split=='train']
    columns=json.loads((processed/'stage3_report.json').read_text())['schemas']['audio']['columns']
    out=processed/'segmentation_study_v1';out.mkdir(exist_ok=True)
    cache=out/'participant_cache';cache.mkdir(exist_ok=True)
    sources={p.relative_to(project).as_posix():sha256(p) for p in [Path(__file__),project/'src/dataset/label_speech_audit.py']}
    kept={};excluded=[]
    def one(pid):
        path=project/'data/raw'/f'{pid}_P.tar.gz';stat=path.stat()
        audit=json.loads((project/'reports/data_audit/label_speech_cache'/f'{pid}.json').read_text())
        key=audit['key'];r=audit['result']
        if key['source']!=sources['src/dataset/label_speech_audit.py'] or key['size']!=stat.st_size or key['mtime_ns']!=stat.st_mtime_ns or not key['features']:
            raise ValueError('Stale timing audit')
        if abs(r['end_gap_seconds'])>1:return int(pid),None
        contract={'sources':sources,'audit':audit,'columns':columns}
        target=cache/f'{pid}.json'
        saved=json.loads(target.read_text()) if target.exists() else {}
        if saved.get('contract')!=contract:
            data=extract(path,columns,r['audio_seconds'])
            if data['member_hashes'][f'{pid}_OpenSMILE2.3.0_egemaps.csv']!=r['egemaps_sha256'] or data['member_hashes'][f'{pid}_Transcript.csv']!=r['transcript_sha256']:
                raise ValueError('Audit content mismatch')
            saved={'contract':contract,'data':data}
            target.write_text(json.dumps(saved,allow_nan=False),encoding='utf-8')
        return int(pid),saved['data']
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,(pid,data) in enumerate(pool.map(one,ids),1):
            if data is None:excluded.append(pid)
            else:kept[pid]=data
            logging.info('Transcript summaries %s/%s',i,len(ids))
    x=pd.DataFrame.from_dict({pid:r['features'] for pid,r in kept.items()},orient='index')
    x.index.name=labels.index.name
    x.to_parquet(out/'features.parquet');labels.loc[x.index,['PHQ_Binary']].to_parquet(out/'labels.parquet')
    manifest={'excluded_ids':excluded,'sources':sources,'partition':'train','speaker_validated':False,
              'coverage':{str(pid):r['coverage'] for pid,r in kept.items()},
              'output_hashes':{n:sha256(out/n) for n in ['features.parquet','labels.parquet']}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return out


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    print(build(Path(__file__).resolve().parents[2]))
