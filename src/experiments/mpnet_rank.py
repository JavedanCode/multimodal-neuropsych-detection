"""Nested compression and cutoff study on frozen MPNet embeddings."""
from datetime import datetime,timezone
import json
import logging
from pathlib import Path
import subprocess
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits
from src.experiments.audio_study import SEEDS,estimator
from src.experiments.baselines import sha256,TARGET_POLICY
from src.experiments.mpnet_study import inputs
from src.experiments.text_temporal_study import summarize
from src.models.text_rank import TextRankPreprocessor


def choose(records,default=False):
    return max([r for r in records if not default or r['threshold']==0],key=lambda r:r['inner_f1'])


def run(project):
    sets,y=inputs(project);x=sets['mpnet']
    out=project/'experiments/runs'/('mpnet_rank_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir();(out/'models').mkdir()
    paths=['docs/mpnet_rank_protocol.md','src/models/text_rank.py','src/models/text_preprocessing.py',
           'src/experiments/mpnet_rank.py','src/experiments/mpnet_study.py','src/experiments/text_temporal_study.py',
           'src/experiments/audio_study.py','src/experiments/metrics.py','src/experiments/baselines.py',
           'data/processed/mpnet_v1/features.parquet','data/processed/mpnet_v1/labels.parquet','data/processed/mpnet_v1/manifest.json']
    hashes={p:sha256(project/p) for p in paths}
    (out/'protocol.json').write_text(json.dumps({'hashes':hashes,'target':TARGET_POLICY,'seeds':SEEDS,
        'participants':list(map(int,y.index)),'test_evaluated':False,'development_evaluated':False},indent=2),encoding='utf-8')
    (out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    predictions=[]
    for seed in SEEDS:
        for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(x,y)):
            fit,valid,target,held=x.iloc[a],x.iloc[b],y.iloc[a],y.iloc[b]
            inner=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(fit,target))
            audit={'fit_ids':list(map(int,target.index)),'heldout_ids':list(map(int,held.index)),
                   'inner_ids':[{'fit':list(map(int,target.index[ia])),'heldout':list(map(int,target.index[ib]))} for ia,ib in inner]}
            candidates=[];selected={}
            for rank in [8,16,32,64]:
                logging.info('MPNet rank seed %s fold %s rank %s',seed,fold,rank)
                prepared=[]
                for ia,ib in inner:
                    prep=TextRankPreprocessor(rank).fit(fit.iloc[ia])
                    prepared.append((ia,ib,prep.transform(fit.iloc[ia]),prep.transform(fit.iloc[ib])))
                records=[]
                for c in [.01,.1,1.,10.]:
                    for weight in [None,'balanced']:
                        params={'C':c,'class_weight':weight};scores=np.full(len(target),np.nan)
                        for ia,ib,f,v in prepared:scores[ib]=estimator('linear',params,seed).fit(f,target.iloc[ia]).decision_function(v)
                        if not np.isfinite(scores).all():raise ValueError('Invalid inner scores')
                        for cutoff in [0.,-.5,.5]:
                            records.append({'rank':rank,'parameters':params,'threshold':cutoff,
                                'inner_f1':float(f1_score(target,scores>=cutoff,zero_division=0)),'inner_scores':scores.tolist()})
                selected[f'rank{rank}_default']=choose(records,True);selected[f'rank{rank}_tuned']=choose(records)
                candidates.extend(records)
            selected['previous']=choose([r for r in candidates if r['rank']==32 and r['parameters']['C']<=1],True)
            selected['selected_default']=choose(candidates,True);selected['selected_tuned']=choose(candidates)
            audit['candidates']=candidates;audit['selected']=selected;fitted={}
            for name,spec in selected.items():
                key=json.dumps({k:spec[k] for k in ['rank','parameters']},sort_keys=True)
                if key not in fitted:
                    fitted[key]=Pipeline([('preprocess',TextRankPreprocessor(spec['rank'])),
                        ('classifier',estimator('linear',spec['parameters'],seed))]).fit(fit,target)
                bundle={'model':fitted[key],'threshold':spec['threshold']}
                raw=bundle['model'].decision_function(valid);scores=raw-spec['threshold']
                checkpoint=out/'models'/f'{name}_{seed}_{fold}.joblib';joblib.dump(bundle,checkpoint)
                restored=joblib.load(checkpoint)
                np.testing.assert_array_equal(scores,restored['model'].decision_function(valid)-restored['threshold'])
                predictions.extend({'condition':name,'scenario':'clean','seed':seed,'fold':fold,'participant_id':int(pid),'target':int(t),
                    'raw_score':float(r),'threshold':spec['threshold'],'score':float(s),'prediction':int(s>=0)}
                    for pid,t,r,s in zip(held.index,held,raw,scores))
            for constant in [0,1]:
                predictions.extend({'condition':f'constant_{constant}','scenario':'clean','seed':seed,'fold':fold,'participant_id':int(pid),
                    'target':int(t),'raw_score':float(constant),'threshold':.5,'score':constant-.5,'prediction':constant} for pid,t in held.items())
            (out/f'fold_{seed}_{fold}.json').write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
    pred=pd.DataFrame(predictions)
    for _,g in pred.groupby(['condition','seed']):
        if len(g)!=len(y) or not g.participant_id.is_unique or set(g.participant_id)!=set(y.index):raise ValueError('Incomplete OOF')
    pred.to_parquet(out/'predictions.parquet',index=False);summarize(pred,out)
    for name,digest in hashes.items():
        if sha256(project/name)!=digest:raise ValueError('Source changed during fitting')
    (out/'verification.json').write_text(json.dumps({'prediction_rows':len(pred),'participants':len(y),'test_evaluated':False,
        'hashes_verified':True,'model_hashes':{p.name:sha256(p) for p in (out/'models').iterdir()}},indent=2),encoding='utf-8')
    return out


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    with threadpool_limits(limits=1):print(run(Path(__file__).resolve().parents[2]))
