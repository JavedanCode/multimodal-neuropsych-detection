"""Matched MiniLM/MPNet evaluation reusing the frozen text classifier recipe."""
from datetime import datetime,timezone
import json
import logging
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.experiments.baselines import sha256,TARGET_POLICY
from src.experiments.audio_study import SEEDS
from src.experiments.text_temporal_study import load,run_text,summarize


def inputs(project):
    old,_,y=load(project);root=project/'data/processed/mpnet_v1'
    m=json.loads((root/'manifest.json').read_text())
    if m['partition']!='train' or m['contract']['source']!=sha256(project/'src/dataset/mpnet.py'):raise ValueError('Encoder provenance mismatch')
    if m['contract']['encoding_source']!=sha256(project/'src/dataset/text_temporal.py'):raise ValueError('Encoding changed')
    if m['contract']['reader_source']!=sha256(project/'src/dataset/lexical.py'):raise ValueError('Reader changed')
    if m['previous_manifest_hash']!=sha256(project/'data/processed/text_temporal_v1/manifest.json'):raise ValueError('Previous inputs changed')
    for name,digest in m['output_hashes'].items():
        if sha256(root/name)!=digest:raise ValueError('Feature table changed')
    pd.testing.assert_series_equal(y,pd.read_parquet(root/'labels.parquet').PHQ_Binary)
    x=pd.read_parquet(root/'features.parquet')
    if not x.index.equals(y.index) or x.shape!=(len(y),768) or not np.isfinite(x.to_numpy()).all():raise ValueError('Invalid encoder features')
    return {'minilm':old['text'],'mpnet':x},y


def run(project):
    sets,y=inputs(project)
    out=project/'experiments/runs'/('mpnet_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir();(out/'models').mkdir()
    paths=['docs/mpnet_protocol.md','src/dataset/mpnet.py','src/dataset/lexical.py','src/dataset/text_temporal.py',
           'src/experiments/mpnet_study.py','src/experiments/text_temporal_study.py','src/models/text_preprocessing.py',
           'src/experiments/audio_study.py','src/experiments/baselines.py','src/experiments/metrics.py',
           'data/processed/mpnet_v1/features.parquet','data/processed/mpnet_v1/labels.parquet','data/processed/mpnet_v1/manifest.json',
           'data/processed/text_temporal_v1/manifest.json','data/processed/text_temporal_v1/sequences.npz','data/processed/labels.parquet']
    hashes={p:sha256(project/p) for p in paths}
    (out/'protocol.json').write_text(json.dumps({'hashes':hashes,'target':TARGET_POLICY,'seeds':SEEDS,
        'participants':list(map(int,y.index)),'test_evaluated':False,'development_evaluated':False},indent=2),encoding='utf-8')
    (out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    pred=run_text(sets,y,out)
    for _,g in pred.groupby(['condition','seed']):
        if len(g)!=len(y) or not g.participant_id.is_unique or set(g.participant_id)!=set(y.index):raise ValueError('Incomplete OOF')
    pred.to_parquet(out/'predictions.parquet',index=False);summarize(pred,out)
    for p,digest in hashes.items():
        if sha256(project/p)!=digest:raise ValueError('Source/input changed during fitting')
    (out/'verification.json').write_text(json.dumps({'prediction_rows':len(pred),'participants':len(y),'hashes_verified':True,
        'model_hashes':{p.name:sha256(p) for p in (out/'models').iterdir()},'test_evaluated':False},indent=2),encoding='utf-8')
    return out


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    with threadpool_limits(limits=1):print(run(Path(__file__).resolve().parents[2]))
