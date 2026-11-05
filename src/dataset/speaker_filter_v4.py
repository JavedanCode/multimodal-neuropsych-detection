"""Conservative text-context prompt estimates; never claims verified speakers."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import re
import numpy as np
import pandas as pd
from src.dataset.lexical import read_transcript
from src.dataset.contracts_v2 import screened_rows
from src.dataset.transcript_context_v4 import chronological
from src.experiments.baselines import sha256
from src.experiments.mpnet_study import inputs

PREFIX=re.compile(r'^(?:(?:and|so|okay|ok|well|now)\s+)*')
DIRECT=re.compile(r"^(?:(?:do|did|does|have|has|had|are|were|will|would|can|could|should)\s+you\b|(?:is|was|does|did)\s+your\b|tell\s+me\s+(?:about|more)\b|(?:can|could|would)\s+you\s+tell\b|what(?:'s|\s+is|\s+was|\s+are|\s+were|\s+would|\s+do|\s+did|\s+has|\s+have)\b.{0,65}\byou(?:r)?\b|how\b.{0,45}\b(?:are|were|do|did|have|has|would|will|can|could)\s+you\b|how\s+(?:is|was)\s+your\b|(?:when|where|why|who)\s+(?:do|did|are|were|have|would|will|can|could)\s+you\b)")
FIRST=re.compile(r"\b(?:i|i'm|i've|i'd|i'll|my|myself)\b")


def normalize(text):
    return re.sub(r'[^a-z0-9\s\']',' ',str(text).lower().replace('\u2019',"'")).strip()


def prompt_like(text):
    s=re.sub(r'\s+',' ',normalize(text));s=PREFIX.sub('',s)
    if len(s.split())>28 or FIRST.search(s):return False
    return bool(DIRECT.search(s))


def masks(frame,rows):
    timeline=chronological(frame);ordered=timeline.source_row.to_numpy(int)
    broad=np.array([prompt_like(t) for t in timeline.Text]);supported=np.zeros(len(timeline),bool)
    for i in np.flatnonzero(broad):
        end=float(timeline.End_Time.iloc[i])
        for j in range(i+1,min(len(timeline),i+4)):
            if float(timeline.Start_Time.iloc[j])-end>30:break
            text=normalize(timeline.Text.iloc[j])
            if not broad[j] and text and (FIRST.search(text) or len(text.split())<=8):supported[i]=True;break
    lookup={int(row):i for i,row in enumerate(ordered)}
    indices=[lookup[int(row)] for row in rows['row_indices']]
    return {'context_filter':~supported[indices],'prompt_filter':~broad[indices]}


def build(project):
    sets,y=inputs(project);root=project/'data/processed';prior=json.loads((root/'mpnet_v1/manifest.json').read_text())
    out=root/'speaker_filter_v4';cache=out/'participant_cache';cache.mkdir(parents=True,exist_ok=True)
    paths=['src/dataset/speaker_filter_v4.py','src/dataset/contracts_v2.py','src/dataset/transcript_context_v4.py',
           'src/dataset/lexical.py','docs/speaker_filter_v4_protocol.md','data/processed/mpnet_v1/manifest.json']
    hashes={p:sha256(project/p) for p in paths}
    def one(pid):
        emb=root/'mpnet_v1/participant_cache'/f'{pid}.npz';assert sha256(emb)==prior['participants'][str(pid)]['hash']
        expected=prior['participants'][str(pid)]['key']['transcript_hash']
        previous_cache=root/'transcript_context_v4/participant_cache'/f'{pid}.parquet'
        if previous_cache.exists():
            meta=json.loads(previous_cache.with_suffix('.json').read_text())
            assert meta['transcript_hash']==expected and meta['cache_hash']==sha256(previous_cache)
            frame=pd.read_parquet(previous_cache)
        else:
            frame,digest=read_transcript(project/'data/raw'/f'{pid}_P.tar.gz');assert digest==expected
        duration=json.loads((project/'reports/data_audit/label_speech_cache'/f'{pid}.json').read_text())['result']['audio_seconds']
        rows=screened_rows(frame,duration);keep=masks(frame,rows)
        with np.load(emb) as z:x=z['embeddings']
        assert len(x)==len(rows['row_indices'])==prior['participants'][str(pid)]['audit']['retained']
        means={};audit={}
        for name,mask in keep.items():
            fallback=not mask.any();means[name]=sets['mpnet'].loc[pid].to_numpy() if fallback else x[mask].mean(0)
            audit[name]={'removed':int((~mask).sum()),'retained':int(mask.sum()),'fallback':bool(fallback)}
        np.savez_compressed(cache/f'{pid}.npz',**keep,row_indices=rows['row_indices'])
        # Restricted transcripts remain local; sufficient for independent replay.
        frame.to_parquet(cache/f'{pid}.parquet',index=False)
        return pid,means,{'transcript_hash':expected,'cache_hash':sha256(cache/f'{pid}.npz'),
                         'frame_hash':sha256(cache/f'{pid}.parquet'),'audit':audit,'utterances':len(x)}
    records={};values={name:[] for name in ['context_filter','prompt_filter']}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,(pid,means,record) in enumerate(pool.map(one,y.index),1):
            records[str(pid)]=record
            for name in values:values[name].append(means[name])
            logging.info('Speaker estimates %s/%s',i,len(y))
    for name,v in values.items():pd.DataFrame(v,index=y.index,columns=sets['mpnet'].columns).to_parquet(out/f'{name}.parquet')
    y.to_frame().to_parquet(out/'labels.parquet')
    for p,h in hashes.items():assert sha256(project/p)==h
    (out/'manifest.json').write_text(json.dumps({'hashes':hashes,'participants':records,'partition':'train','speaker_verified':False,
        'outputs':{p.name:sha256(p) for p in out.glob('*.parquet')}},indent=2));return out


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(message)s');print(build(Path(__file__).resolve().parents[2]))
