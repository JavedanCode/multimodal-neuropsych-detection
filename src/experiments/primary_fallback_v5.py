"""Nested audio/facial primary study with explicit text-fallback routing."""
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
from src.experiments.aligned_fusion_v2 import load
from src.experiments.baselines import sha256
from src.experiments.audio_study import SEEDS
from src.experiments.domain_v2 import splits
from src.experiments.text_temporal_study import save_predictions,summarize
from src.experiments.compare_text_temporal import paired
from src.models.quality_stack_v2 import take,crossfit,base_fit,base_scores
from src.models.primary_fallback_v5 import fit_meta,from_scores,predict,SCENARIOS

CONDITIONS={'primary_plain':(False,False),'primary_quality':(True,False),
            'availability_fallback':(False,True),'quality_fallback':(True,True)}


def inner_candidates(data,y,seed,fold):
    inner=splits(y,3,seed+fold+1);prepared=[];records=[]
    for j,(a,b) in enumerate(inner):
        train,valid=take(data,a),take(data,b)
        scores,record=crossfit(train,y.iloc[a],seed+fold+j+100)
        base=base_fit(train,y.iloc[a],seed)
        prepared.append((a,b,train,valid,scores,base_scores(base,valid)));records.append(record)
    candidates=[]
    for name,(quality,fallback) in CONDITIONS.items():
        for c in [.01,.1,1.]:
            scores=np.full(len(y),np.nan);routes=np.zeros(len(y),bool)
            for a,b,train,valid,fit_scores,held_scores in prepared:
                meta=fit_meta(fit_scores,train,y.iloc[a],quality,c,seed)
                scores[b],routes[b]=from_scores(meta,held_scores,valid,quality,fallback)
            candidates.append({'condition':name,'C':c,'scores':scores.tolist(),'routes':routes.tolist(),'inner_f1':float(f1_score(y,scores>=0))})
    selected={name:max([r for r in candidates if r['condition']==name],key=lambda r:r['inner_f1']) for name in CONDITIONS}
    return selected,{'candidates':candidates,'selected':selected,'crossfit':records,
        'inner_ids':[{'fit':list(map(int,y.index[a])),'heldout':list(map(int,y.index[b]))} for a,b in inner]}


def run(project,baseline):
    baseline=baseline.resolve();data,y,_=load(project);bm=json.loads((baseline/'manifest.json').read_text())
    for p,h in bm['hashes'].items():assert sha256(project/p)==h
    paths=['docs/primary_fallback_v5_protocol.md','src/models/primary_fallback_v5.py','src/experiments/primary_fallback_v5.py',
           'src/models/quality_stack_v2.py','src/models/bounded_v3.py','src/experiments/domain_v2.py',
           'src/experiments/aligned_fusion_v2.py','src/models/text_preprocessing.py','src/experiments/text_temporal_study.py',
           'data/processed/aligned_v2/manifest.json','data/processed/mpnet_v1/features.parquet']
    hashes={p:sha256(project/p) for p in paths}
    for p in (baseline/'models').glob('joint_*.joblib'):hashes[str(p.relative_to(project))]=sha256(p)
    out=project/'experiments/runs'/('primary_fallback_v5_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir();(out/'models').mkdir()
    (out/'protocol.json').write_text(json.dumps({'hashes':hashes,'test_evaluated':False},indent=2))
    predictions=[];aps=[]
    for seed in SEEDS:
        for fold,(a,b) in enumerate(splits(y,5,seed)):
            logging.info('Primary fallback seed%s fold%s',seed,fold)
            target,held=y.iloc[a],y.iloc[b];train,valid=take(data,a),take(data,b)
            selected,audit=inner_candidates(train,target,seed,fold)
            scores,outer=crossfit(train,target,seed+fold+200);base=base_fit(train,target,seed)
            bundles={name:{'kind':'primary','quality':quality,'fallback':fallback,'base':base,
                    'meta':fit_meta(scores,train,target,quality,selected[name]['C'],seed)} for name,(quality,fallback) in CONDITIONS.items()}
            bundles['fixed_text']={'kind':'text','model':base['text']}
            bundles['retained_reference']={'kind':'text','model':joblib.load(baseline/'models'/f'joint_{seed}_{fold}.joblib')}
            for name,bundle in bundles.items():
                joblib.dump(bundle,out/'models'/f'{name}_{seed}_{fold}.joblib')
                for scenario in SCENARIOS:
                    if bundle['kind']=='text':values=bundle['model'].decision_function(valid['text']);route=np.zeros(len(held),bool)
                    else:values,route=predict(bundle,valid,scenario)
                    save_predictions(predictions,name,scenario,seed,fold,held,values)
                    for row,r in zip(predictions[-len(held):],route):row.update(fallback_used=bool(r),text_used=bool(r or bundle['kind']=='text'))
                    aps.append({'condition':name,'scenario':scenario,'seed':seed,'fold':fold,'average_precision':average_precision_score(held,values)})
            audit.update(fit_ids=list(map(int,target.index)),heldout_ids=list(map(int,held.index)),outer_crossfit=outer)
            (out/f'fold_{seed}_{fold}.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
    pred=pd.DataFrame(predictions);pred.to_parquet(out/'predictions.parquet',index=False);metrics=summarize(pred,out)
    pd.DataFrame(aps).to_csv(out/'fold_average_precision.csv',index=False)
    pred.groupby(['condition','scenario','seed'])[['fallback_used','text_used']].mean().to_csv(out/'routing.csv')
    comparison={'against_reference':[paired(metrics,n,'retained_reference') for n in CONDITIONS],
                'fallback_ablation':[paired(metrics,n,c,s) for n,c in [('availability_fallback','primary_plain'),('quality_fallback','primary_quality')] for s in SCENARIOS]}
    (out/'comparison.json').write_text(json.dumps(comparison,indent=2))
    for p,h in hashes.items():assert sha256(project/p)==h
    (out/'manifest.json').write_text(json.dumps({'hashes':hashes,'baseline':str(baseline),'models':{p.name:sha256(p) for p in (out/'models').iterdir()},
        'test_evaluated':False,'development_evaluated':False},indent=2));return out


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    with threadpool_limits(limits=1):print(run(Path(__file__).resolve().parents[2],Path(sys.argv[1])))
