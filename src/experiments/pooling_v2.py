"""Domain-balanced nested learning-curve selection, never outer early stopping."""
from pathlib import Path
from datetime import datetime,timezone
import json
import logging
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,average_precision_score
from threadpoolctl import threadpool_limits
from src.experiments.utterance_pooling import load,configure
from src.experiments.domain_v2 import splits
from src.experiments.audio_study import SEEDS
from src.experiments.baselines import sha256
from src.experiments.text_temporal_study import save_predictions,summarize
from src.experiments.compare_text_temporal import paired
from src.models.utterance_pooling import scale_fit,scale_transform
from src.models.pooling_v2 import train,predict,EPOCHS


def run(project,baseline):
    baseline=baseline.resolve()
    configure();features,x,mask,y,cache_hashes=load(project)
    bm=json.loads((baseline/'manifest.json').read_text())
    for p,h in bm['hashes'].items():assert sha256(project/p)==h
    out=project/'experiments/runs'/('pooling_v2_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir();(out/'models').mkdir()
    paths=['src/models/pooling_v2.py','src/experiments/pooling_v2.py','src/models/utterance_pooling.py',
           'src/experiments/domain_v2.py','docs/pipeline_v2_protocol.md']
    hashes={p:sha256(project/p) for p in paths};hashes.update(cache_hashes)
    for p in (baseline/'models').glob('joint_*.joblib'):hashes[str(p.relative_to(project))]=sha256(p)
    records=[];curves=[];aps=[]
    for seed in SEEDS:
        for fold,(a,b) in enumerate(splits(y,5,seed)):
            logging.info('Pooling v2 seed%s fold%s',seed,fold);target=y.iloc[a];held=y.iloc[b];inner=splits(target,3,seed+fold+1)
            candidates=[]
            for attention in [False,True]:
                for decay in [.01,.1]:
                    scores={e:np.full(len(a),np.nan) for e in EPOCHS}
                    for j,(ia,ib) in enumerate(inner):
                        stats=scale_fit(x[a[ia]],mask[a[ia]]);z=scale_transform(x[a[ia]],mask[a[ia]],stats);v=scale_transform(x[a[ib]],mask[a[ib]],stats)
                        _,values,history=train(z,mask[a[ia]],target.iloc[ia],attention,seed+fold,decay,100,(v,mask[a[ib]],target.iloc[ib]))
                        curves.extend({'seed':seed,'fold':fold,'inner_fold':j,'attention':attention,'decay':decay,**r} for r in history)
                        for e,s in values.items():scores[e][ib]=s
                    candidates.extend({'attention':attention,'decay':decay,'epochs':e,'scores':s.tolist(),'inner_f1':float(f1_score(target,s>=0))} for e,s in scores.items())
            selected={name:max([r for r in candidates if r['attention']==att],key=lambda r:r['inner_f1']) for name,att in [('mean',False),('attention',True)]}
            selected['selected']=max(candidates,key=lambda r:r['inner_f1']);stats=scale_fit(x[a],mask[a]);z=scale_transform(x[a],mask[a],stats);fitted={}
            for name,spec in selected.items():
                key=(spec['attention'],spec['decay'],spec['epochs'])
                if key not in fitted:
                    state,_,history=train(z,mask[a],target,spec['attention'],seed+fold,spec['decay'],spec['epochs'])
                    fitted[key]={'state':state,'attention':spec['attention'],'decay':spec['decay'],'epochs':spec['epochs'],'stats':stats,'training_history':history}
                bundle=fitted[key];joblib.dump(bundle,out/'models'/f'{name}_{seed}_{fold}.joblib');scores=predict(bundle,x[b],mask[b])
                save_predictions(records,name,'clean',seed,fold,held,scores)
                aps.append({'condition':name,'seed':seed,'fold':fold,'average_precision':average_precision_score(held,scores)})
            model=joblib.load(baseline/'models'/f'joint_{seed}_{fold}.joblib');scores=model.decision_function(features.iloc[b])
            save_predictions(records,'reference','clean',seed,fold,held,scores)
            aps.append({'condition':'reference','seed':seed,'fold':fold,'average_precision':average_precision_score(held,scores)})
            audit={'fit_ids':list(map(int,target.index)),'heldout_ids':list(map(int,held.index)),
                   'inner_ids':[{'fit':list(map(int,target.index[ia])),'heldout':list(map(int,target.index[ib]))} for ia,ib in inner],
                   'candidates':candidates,'selected':selected}
            (out/f'fold_{seed}_{fold}.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
    pred=pd.DataFrame(records);pred.to_parquet(out/'predictions.parquet',index=False);metrics=summarize(pred,out)
    pd.DataFrame(curves).to_csv(out/'inner_learning_curves.csv',index=False);pd.DataFrame(aps).to_csv(out/'fold_average_precision.csv',index=False)
    (out/'comparison.json').write_text(json.dumps([paired(metrics,n,'reference') for n in ['mean','attention','selected']],indent=2))
    for p,h in hashes.items():assert sha256(project/p)==h
    (out/'manifest.json').write_text(json.dumps({'hashes':hashes,'baseline':str(baseline),'model_hashes':{p.name:sha256(p) for p in (out/'models').iterdir()},'test_evaluated':False},indent=2))
    return out


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    with threadpool_limits(limits=1):print(run(Path(__file__).resolve().parents[2],Path(sys.argv[1])))
