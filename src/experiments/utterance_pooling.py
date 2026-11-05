"""Training-only nested learned pooling comparison."""
from datetime import datetime,timezone
from pathlib import Path
import json
import logging
import sys
import subprocess
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from src.experiments.audio_study import SEEDS
from src.experiments.baselines import sha256
from src.experiments.mpnet_study import inputs
from src.experiments.text_temporal_study import summarize,save_predictions
from src.experiments.compare_text_temporal import paired
from src.models.utterance_pooling import scale_fit,scale_transform,train,predict


def configure():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')


def load(project):
    sets,y=inputs(project);root=project/'data/processed/mpnet_v1'
    manifest=json.loads((root/'manifest.json').read_text());bags=[];hashes={}
    projection=np.random.default_rng(42).normal(size=(768,64)).astype(np.float32)/np.sqrt(np.float32(64))
    for pid in y.index:
        path=root/'participant_cache'/f'{pid}.npz';digest=sha256(path)
        assert digest==manifest['participants'][str(pid)]['hash']
        hashes[str(path.relative_to(project))]=digest
        with np.load(path) as data:
            e=data['embeddings'];assert e.ndim==2 and e.shape[1]==768 and np.isfinite(e).all()
            np.testing.assert_array_equal(e.mean(0),data['mean'])
            np.testing.assert_array_equal(data['mean'],sets['mpnet'].loc[pid].to_numpy())
            bags.append(e@projection)
    x=np.zeros((len(y),max(map(len,bags)),64),dtype=np.float32);mask=np.zeros(x.shape[:2],bool)
    for i,bag in enumerate(bags):x[i,:len(bag)]=bag;mask[i,:len(bag)]=True
    return sets['mpnet'],x,mask,y,hashes


def run(project):
    configure();features,x,mask,y,cache_hashes=load(project)
    out=project/'experiments/runs'/('utterance_pooling_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir();(out/'models').mkdir()
    paths=['docs/utterance_pooling_protocol.md','src/models/utterance_pooling.py','src/experiments/utterance_pooling.py',
           'src/experiments/mpnet_study.py','src/experiments/text_temporal_study.py','src/experiments/metrics.py',
           'src/experiments/compare_text_temporal.py','data/processed/mpnet_v1/manifest.json']
    hashes={p:sha256(project/p) for p in paths};hashes.update(cache_hashes)
    old=project/'experiments/runs/mpnet_20261102T171156498575Z'
    hashes[str((old/'predictions.parquet').relative_to(project))]=sha256(old/'predictions.parquet')
    for p in (old/'models').glob('mpnet_linear_*.joblib'):hashes[str(p.relative_to(project))]=sha256(p)
    (out/'protocol.json').write_text(json.dumps({'hashes':hashes,'participants':list(map(int,y.index)),
        'test_evaluated':False,'development_evaluated':False,'device':torch.cuda.get_device_name(0)},indent=2))
    (out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    predictions=[]
    for seed in SEEDS:
        for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(features,y)):
            logging.info('Pooling seed %s fold %s',seed,fold)
            target=y.iloc[a];held=y.iloc[b]
            splits=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(a,target))
            scores={(attention,epoch):np.full(len(a),np.nan) for attention in (False,True) for epoch in (40,100)}
            for ia,ib in splits:
                stats=scale_fit(x[a[ia]],mask[a[ia]])
                z=scale_transform(x[a[ia]],mask[a[ia]],stats);v=scale_transform(x[a[ib]],mask[a[ib]],stats)
                for attention in (False,True):
                    _,values=train(z,mask[a[ia]],target.iloc[ia],attention,seed+fold,100,(v,mask[a[ib]]))
                    for epoch,values_ in values.items():scores[attention,epoch][ib]=values_
            records=[{'attention':attention,'epochs':epoch,'inner_f1':float(f1_score(target,values>=0)),
                      'scores':values.tolist()} for (attention,epoch),values in scores.items()]
            selected={name:max([r for r in records if r['attention']==attention],key=lambda r:r['inner_f1'])
                      for name,attention in [('mean',False),('attention',True)]}
            selected['selected']=max(records,key=lambda r:r['inner_f1'])
            stats=scale_fit(x[a],mask[a]);z=scale_transform(x[a],mask[a],stats);fitted={}
            for name,spec in selected.items():
                key=(spec['attention'],spec['epochs'])
                if key not in fitted:
                    state,_=train(z,mask[a],target,spec['attention'],seed+fold,spec['epochs'])
                    fitted[key]={'state':state,'stats':stats,'attention':spec['attention'],'epochs':spec['epochs']}
                bundle=fitted[key];joblib.dump(bundle,out/'models'/f'{name}_{seed}_{fold}.joblib')
                save_predictions(predictions,name,'clean',seed,fold,held,predict(bundle,x[b],mask[b]))
            base=joblib.load(old/'models'/f'mpnet_linear_{seed}_{fold}.joblib')
            save_predictions(predictions,'reference','clean',seed,fold,held,base.decision_function(features.iloc[b]))
            audit={'fit_ids':list(map(int,target.index)),'heldout_ids':list(map(int,held.index)),
                   'inner_ids':[{'fit':list(map(int,target.index[ia])),'heldout':list(map(int,target.index[ib]))} for ia,ib in splits],
                   'candidates':records,'selected':selected}
            (out/f'fold_{seed}_{fold}.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
    pred=pd.DataFrame(predictions);pred.to_parquet(out/'predictions.parquet',index=False);metrics=summarize(pred,out)
    comparisons=[paired(metrics,name,'reference') for name in ['mean','attention','selected']]
    comparisons.append(paired(metrics,'attention','mean'))
    (out/'comparison.json').write_text(json.dumps(comparisons,indent=2))
    for p,h in hashes.items():assert sha256(project/p)==h
    (out/'verification.json').write_text(json.dumps({'model_hashes':{p.name:sha256(p) for p in (out/'models').iterdir()},'test_evaluated':False},indent=2))
    return out


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    with threadpool_limits(limits=1):print(run(Path(__file__).resolve().parents[2]))
