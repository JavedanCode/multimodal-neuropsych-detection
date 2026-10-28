"""Bounded richer eGeMAPS summaries for official training participants only."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import logging
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd

from src.dataset.inventory import _is_unsafe_member
from src.experiments.baselines import sha256

MAX_BYTES=256*1024*1024
MAX_ROWS=500000
WINDOW_SECONDS=10


def summarize(frame, columns):
    if not set(columns+['frameTime']).issubset(frame.columns) or not len(frame) or len(frame)>MAX_ROWS:
        raise ValueError('Invalid audio schema or row count')
    times=pd.to_numeric(frame['frameTime'],errors='coerce').to_numpy(float)
    if not np.isfinite(times).all() or (times<0).any() or (np.diff(times)<0).any():
        raise ValueError('Finite nonnegative monotonic frameTime required')
    windows=np.floor(times/WINDOW_SECONDS).astype(np.int64)
    result={}
    for column in columns:
        values=pd.to_numeric(frame[column],errors='coerce').to_numpy(float)
        finite=np.isfinite(values);valid=values[finite]
        stats={s:None for s in ['mean','std','p10','p25','p50','p75','p90','iqr','adjacent_change','window_mean_std']}
        if len(valid):
            q=np.quantile(valid,[.1,.25,.5,.75,.9],method='linear')
            both=finite[:-1]&finite[1:]
            delta=np.abs(values[1:][both]-values[:-1][both])
            means=pd.Series(valid).groupby(windows[finite]).mean().to_numpy()
            stats.update(mean=float(valid.mean()),std=float(valid.std(ddof=0)),
                **{k:float(v) for k,v in zip(['p10','p25','p50','p75','p90'],q)},
                iqr=float(q[3]-q[1]),adjacent_change=float(delta.mean()) if len(delta) else None,
                window_mean_std=float(means.std(ddof=0)))
        result.update({f'audio__{column}__{stat}':value for stat,value in stats.items()})
    return result


def extract(path, columns):
    wanted=f'{int(path.name.split("_")[0])}_OpenSMILE2.3.0_egemaps.csv'
    with tarfile.open(path,'r|gz') as archive:
        for member in archive:
            if _is_unsafe_member(member.name) or member.issym() or member.islnk():
                raise ValueError('Unsafe archive member')
            if Path(member.name).name!=wanted:continue
            if not member.isfile() or member.size>MAX_BYTES:raise ValueError('Invalid/oversized audio member')
            raw=archive.extractfile(member).read(MAX_BYTES+1)
            if len(raw)>MAX_BYTES:raise ValueError('Audio CSV exceeds memory bound')
            frame=pd.read_csv(io.BytesIO(raw),sep=';',nrows=MAX_ROWS+1)
            frame.columns=frame.columns.str.strip()
            if frame.columns.duplicated().any():raise ValueError('Duplicate audio columns')
            return {'features':summarize(frame,columns),'rows':len(frame),
                    'member_sha256':hashlib.sha256(raw).hexdigest(),'member':member.name}
    raise ValueError('Missing eGeMAPS CSV')


def build(project):
    processed=project/'data/processed'
    labels=pd.read_parquet(processed/'labels.parquet')
    if not labels.index.is_unique or set(labels.columns)!={'split','PHQ_Binary'}:
        raise ValueError('Invalid label table')
    ids=labels.index[labels.split=='train']
    schema=json.loads((processed/'stage3_report.json').read_text())['schemas']['audio']['columns']
    raw_root=project/'data/raw'
    output=processed/'audio_study_v1';output.mkdir(exist_ok=True)
    cache=output/'participant_cache';cache.mkdir(exist_ok=True)
    contract={'source':sha256(Path(__file__)),'schema':schema,'window_seconds':WINDOW_SECONDS,
              'max_bytes':MAX_BYTES,'max_rows':MAX_ROWS}
    def one(pid):
        path=raw_root/f'{pid}_P.tar.gz';stat=path.stat()
        key={**contract,'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'root':str(raw_root.resolve())}
        target=cache/f'{pid}.json'
        if target.exists():
            saved=json.loads(target.read_text())
            if saved['key']==key:return int(pid),saved
        saved={'key':key,'data':extract(path,schema)}
        temporary=target.with_suffix('.tmp')
        temporary.write_text(json.dumps(saved,allow_nan=False),encoding='utf-8');temporary.replace(target)
        return int(pid),saved
    rows={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,(pid,saved) in enumerate(pool.map(one,ids),1):
            rows[pid]=saved
            logging.info('Audio summaries: %d/%d training participants',i,len(ids))
    features=pd.DataFrame.from_dict({p:r['data']['features'] for p,r in rows.items()},orient='index').reindex(ids)
    features.index.name='Participant_ID'
    existing=pd.read_parquet(processed/'features.parquet').loc[ids]
    base=[c for c in existing if c.startswith('audio__')]
    np.testing.assert_allclose(features[base],existing[base],rtol=1e-8,atol=1e-8,equal_nan=True)
    features.to_parquet(output/'features.parquet')
    labels.loc[ids,['PHQ_Binary']].to_parquet(output/'labels.parquet')
    manifest={'contract':contract,'participant_ids':list(map(int,ids)),'partition':'train',
        'baseline_columns':base,'participants':{str(p):{k:v for k,v in r['data'].items() if k!='features'} for p,r in rows.items()},
        'original_input_sha256':{n:sha256(processed/n) for n in ['features.parquet','labels.parquet','stage3_report.json']},
        'output_sha256':{n:sha256(output/n) for n in ['features.parquet','labels.parquet']},
        'baseline_summaries_verified':True,'test_evaluated':False}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return output


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    print(build(Path(__file__).resolve().parents[2]))
