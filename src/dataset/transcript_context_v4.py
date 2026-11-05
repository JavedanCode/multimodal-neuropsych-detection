"""Extract chronological transcript context for estimated speaker-role review."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
import pandas as pd
from src.dataset.lexical import read_transcript
from src.experiments.baselines import sha256


def chronological(frame):
    f=frame.copy();f['source_row']=np.arange(len(f))
    for name in ['Start_Time','End_Time']:f[name]=pd.to_numeric(f[name],errors='coerce')
    f['Text']=f.Text.fillna('').astype(str)
    f=f[np.isfinite(f.Start_Time)&np.isfinite(f.End_Time)&(f.End_Time>f.Start_Time)&(f.Start_Time>=0)]
    return f.sort_values('Start_Time',kind='stable').reset_index(drop=True)


def context(frame,start,end,radius=3):
    f=chronological(frame)
    indices=np.flatnonzero(np.isclose(f.Start_Time,start,rtol=0,atol=1e-8)&np.isclose(f.End_Time,end,rtol=0,atol=1e-8))
    if len(indices)!=1:raise ValueError('Target must match exactly one full transcript interval')
    i=int(indices[0]);rows=[]
    for j in range(max(0,i-radius),min(len(f),i+radius+1)):
        r=f.iloc[j]
        rows.append({'target':j==i,'source_row':int(r.source_row),'start':float(r.Start_Time),
                     'end':float(r.End_Time),'text':r.Text})
    return rows


def build(project):
    source=project/'data/processed/aligned_v2/speaker_review/context_v3/clips.json'
    clips=json.loads(source.read_text(encoding='utf-8'));prior=json.loads((project/'data/processed/mpnet_v1/manifest.json').read_text())
    out=project/'data/processed/transcript_context_v4';cache=out/'participant_cache';cache.mkdir(parents=True,exist_ok=True)
    ids=sorted({int(r['file'].split('_')[0]) for r in clips})
    def one(pid):
        path=cache/f'{pid}.parquet';meta=cache/f'{pid}.json';expected=prior['participants'][str(pid)]['key']['transcript_hash']
        saved=json.loads(meta.read_text()) if meta.exists() else {}
        if path.exists() and saved.get('transcript_hash')==expected and saved.get('cache_hash')==sha256(path):
            return pid,pd.read_parquet(path)
        frame,digest=read_transcript(project/'data/raw'/f'{pid}_P.tar.gz')
        if digest!=expected:raise ValueError('Source transcript changed')
        frame.to_parquet(path,index=False)
        meta.write_text(json.dumps({'transcript_hash':digest,'cache_hash':sha256(path)},indent=2))
        return pid,frame
    with ThreadPoolExecutor(max_workers=2) as pool:frames=dict(pool.map(one,ids))
    cases=[{'case':r['file'],'participant_id':int(r['file'].split('_')[0]),
            'rows':context(frames[int(r['file'].split('_')[0])],r['start'],r['end'])} for r in clips]
    (out/'contexts.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    manifest={'scope':'Eleven flagged pilot utterances only; no questionnaire labels or model predictions provided',
              'source_hash':sha256(Path(__file__)),'clip_manifest_hash':sha256(source),
              'contexts_hash':sha256(out/'contexts.json'),'caches':{p.name:sha256(p) for p in cache.glob('*.parquet')},
              'speaker_verified':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));return out


if __name__=='__main__':print(build(Path(__file__).resolve().parents[2]))
