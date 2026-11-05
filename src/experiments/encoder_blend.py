"""Nested combination of complementary frozen sentence encoders."""
from datetime import datetime,timezone
from pathlib import Path
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from src.experiments.audio_study import SEEDS,candidates,estimator
from src.experiments.baselines import sha256
from src.experiments.mpnet_study import inputs
from src.experiments.text_temporal_study import summarize,save_predictions
from src.experiments.compare_text_temporal import paired
from src.models.text_preprocessing import TextPreprocessor


def combinations(scores,y):
    records=[]
    for w in [1.,.75,.5,.25,0.]:
        for i in range(len(scores['mpnet']) if w else 1):
            for j in range(len(scores['minilm']) if w!=1 else 1):
                s=w*np.array(scores['mpnet'][i]['scores'])+(1-w)*np.array(scores['minilm'][j]['scores'])
                records.append({'weight':w,'mpnet':i,'minilm':j,'inner_f1':float(f1_score(y,s>=0))})
    return records


def run(project):
    sets,y=inputs(project)
    out=project/'experiments/runs'/('encoder_blend_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir();(out/'models').mkdir()
    paths=['docs/encoder_blend_protocol.md','src/experiments/encoder_blend.py',
           'src/experiments/mpnet_study.py','src/experiments/audio_study.py',
           'src/models/text_preprocessing.py','src/experiments/text_temporal_study.py',
           'src/experiments/metrics.py','src/experiments/compare_text_temporal.py',
           'data/processed/mpnet_v1/features.parquet','data/processed/mpnet_v1/labels.parquet',
           'data/processed/text_temporal_v1/sequences.npz']
    hashes={p:sha256(project/p) for p in paths}
    (out/'protocol.json').write_text(json.dumps({'hashes':hashes,'test_evaluated':False,'development_evaluated':False},indent=2))
    pred=[]
    for seed in SEEDS:
        for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(sets['mpnet'],y)):
            logging.info('Encoder blend seed %s fold %s',seed,fold)
            target,held=y.iloc[a],y.iloc[b]
            splits=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(a,target))
            inner={}
            for name,x in sets.items():
                prepared=[]
                for ia,ib in splits:
                    prep=TextPreprocessor().fit(x.iloc[a[ia]])
                    prepared.append((ia,ib,prep.transform(x.iloc[a[ia]]),prep.transform(x.iloc[a[ib]])))
                inner[name]=[]
                for params in candidates('linear'):
                    scores=np.full(len(a),np.nan)
                    for ia,ib,z,v in prepared:scores[ib]=estimator('linear',params,seed).fit(z,target.iloc[ia]).decision_function(v)
                    inner[name].append({'parameters':params,'scores':scores.tolist()})
            records=combinations(inner,target)
            selected=max(records,key=lambda r:r['inner_f1'])
            reference=max([r for r in records if r['weight']==1],key=lambda r:r['inner_f1'])
            for name,spec in [('reference',reference),('blend',selected)]:
                models={};scores=[]
                for encoder,w in [('mpnet',spec['weight']),('minilm',1-spec['weight'])]:
                    if w==0:continue
                    params=inner[encoder][spec[encoder]]['parameters'];x=sets[encoder]
                    model=Pipeline([('preprocess',TextPreprocessor()),('classifier',estimator('linear',params,seed))]).fit(x.iloc[a],target)
                    models[encoder]=model;scores.append(w*model.decision_function(x.iloc[b]))
                bundle={'models':models,'weight':spec['weight']}
                joblib.dump(bundle,out/'models'/f'{name}_{seed}_{fold}.joblib')
                save_predictions(pred,name,'clean',seed,fold,held,np.sum(scores,axis=0))
            audit={'fit_ids':list(map(int,target.index)),'heldout_ids':list(map(int,held.index)),
                   'inner_ids':[{'fit':list(map(int,target.index[ia])),'heldout':list(map(int,target.index[ib]))} for ia,ib in splits],
                   'inner_scores':inner,'candidates':records,'reference':reference,'selected':selected}
            (out/f'fold_{seed}_{fold}.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
    pred=pd.DataFrame(pred);pred.to_parquet(out/'predictions.parquet',index=False);metrics=summarize(pred,out)
    (out/'comparison.json').write_text(json.dumps(paired(metrics,'blend','reference'),indent=2))
    for p,h in hashes.items():assert sha256(project/p)==h
    (out/'verification.json').write_text(json.dumps({'model_hashes':{p.name:sha256(p) for p in (out/'models').iterdir()},'test_evaluated':False},indent=2))
    return out


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    with threadpool_limits(limits=1):print(run(Path(__file__).resolve().parents[2]))
