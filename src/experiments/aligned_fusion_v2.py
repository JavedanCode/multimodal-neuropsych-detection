"""Nested comparison of text, ordinary and measured-quality stacking."""
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
from src.experiments.mpnet_study import inputs
from src.experiments.domain_v2 import splits
from src.experiments.audio_study import SEEDS
from src.experiments.baselines import sha256
from src.experiments.text_temporal_study import save_predictions,summarize
from src.experiments.compare_text_temporal import paired
from src.models.quality_stack_v2 import take,crossfit,base_fit,base_scores,design,fit_meta,predict,SCENARIOS


def load(project):
    sets,y=inputs(project);root=project/'data/processed/aligned_v2';m=json.loads((root/'manifest.json').read_text())
    assert m['partition']=='train' and set(m['participants'])==set(map(str,y.index))
    for p,h in m['sources'].items():assert sha256(project/p)==h
    assert sha256(root/'labels.parquet')==m['labels_hash']
    pd.testing.assert_series_equal(y,pd.read_parquet(root/'labels.parquet').PHQ_Binary)
    values={'audio':[],'face':[],'quality':[],'available':[]}
    for pid in y.index:
        path=root/'participant_cache'/f'{pid}.npz';assert sha256(path)==m['participants'][str(pid)]['hash']
        with np.load(path) as z:
            for i,name in [(1,'audio'),(2,'face')]:
                a=z[name][z['masks'][:,i]]
                values[name].append(np.r_[a.mean(0),a.std(0)] if len(a) else np.zeros(2*z[name].shape[1]))
            values['quality'].append(z['quality'].mean(0));values['available'].append(z['masks'].any(0))
    data={k:np.asarray(v,dtype=float) for k,v in values.items()};data['text']=sets['mpnet']
    return data,y,m


def run(project,baseline):
    baseline=baseline.resolve();data,y,manifest=load(project)
    out=project/'experiments/runs'/('aligned_fusion_v2_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir();(out/'models').mkdir();predictions=[];aps=[]
    sources=['src/experiments/aligned_fusion_v2.py','src/models/quality_stack_v2.py','src/experiments/domain_v2.py',
             'src/dataset/contracts_v2.py','src/models/text_preprocessing.py','docs/aligned_fusion_v2_protocol.md',
             'data/processed/aligned_v2/manifest.json','data/processed/mpnet_v1/features.parquet']
    hashes={p:sha256(project/p) for p in sources}
    bm=json.loads((baseline/'manifest.json').read_text())
    for p,h in bm['hashes'].items():assert sha256(project/p)==h
    for p in (baseline/'models').glob('joint_*.joblib'):hashes[str(p.relative_to(project))]=sha256(p)
    for seed in SEEDS:
        for fold,(a,b) in enumerate(splits(y,5,seed)):
            logging.info('Aligned fusion seed%s fold%s',seed,fold)
            target,held=y.iloc[a],y.iloc[b];fit,valid=take(data,a),take(data,b);inner=splits(target,3,seed+fold+1)
            prepared=[];crossfit_records=[]
            for j,(ia,ib) in enumerate(inner):
                train,validate=take(fit,ia),take(fit,ib)
                scores,records=crossfit(train,target.iloc[ia],seed+fold+j+100)
                models=base_fit(train,target.iloc[ia],seed);vs=base_scores(models,validate)
                prepared.append((ia,ib,train,validate,scores,vs));crossfit_records.append(records)
            candidates=[]
            for kind in ['text','ordinary','quality']:
                for c in [.01,.1,1.]:
                    scores=np.full(len(target),np.nan)
                    for ia,ib,train,validate,s,vs in prepared:
                        model=fit_meta(s,train,target.iloc[ia],kind,c,seed)
                        scores[ib]=model.decision_function(design(vs,validate['quality'],validate['available'],kind))
                    candidates.append({'kind':kind,'C':c,'scores':scores.tolist(),'inner_f1':float(f1_score(target,scores>=0))})
            selected={k:max([r for r in candidates if r['kind']==k],key=lambda r:r['inner_f1']) for k in ['text','ordinary','quality']}
            oldaudit=json.loads((baseline/f'joint_{seed}_{fold}.json').read_text())
            ref={'kind':'reference','inner_f1':oldaudit['selected']['pooled_f1']}
            selected['selected']=max([ref]+candidates,key=lambda r:r['inner_f1'])
            scores,outercross=crossfit(fit,target,seed+fold+200);base=base_fit(fit,target,seed);reference=joblib.load(baseline/'models'/f'joint_{seed}_{fold}.joblib')
            selected['reference']=ref
            for name,spec in selected.items():
                bundle={'kind':'reference','model':reference} if spec['kind']=='reference' else {'kind':spec['kind'],'C':spec['C'],'base':base,'meta':fit_meta(scores,fit,target,spec['kind'],spec['C'],seed)}
                joblib.dump(bundle,out/'models'/f'{name}_{seed}_{fold}.joblib')
                for scenario in SCENARIOS:
                    result=reference.decision_function(valid['text']) if bundle['kind']=='reference' else predict(bundle,valid,scenario)
                    save_predictions(predictions,name,scenario,seed,fold,held,result)
                    aps.append({'condition':name,'scenario':scenario,'seed':seed,'fold':fold,'average_precision':average_precision_score(held,result)})
            audit={'fit_ids':list(map(int,target.index)),'heldout_ids':list(map(int,held.index)),
                   'inner_ids':[{'fit':list(map(int,target.index[ia])),'heldout':list(map(int,target.index[ib]))} for ia,ib in inner],
                   'inner_crossfit':crossfit_records,'outer_crossfit':outercross,'candidates':candidates,'selected':selected}
            (out/f'fold_{seed}_{fold}.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
    pred=pd.DataFrame(predictions);pred.to_parquet(out/'predictions.parquet',index=False);metrics=summarize(pred,out)
    pd.DataFrame(aps).to_csv(out/'fold_average_precision.csv',index=False)
    clean=[paired(metrics,n,'reference') for n in ['text','ordinary','quality','selected']]
    quality=[paired(metrics,'quality','ordinary',s) for s in SCENARIOS]
    gain=float(np.mean([r['mean_differences']['f1'] for r in quality[1:]]))
    (out/'comparison.json').write_text(json.dumps({'clean':clean,'quality':quality,'quality_missing_gain':gain,
        'passes_robustness_screen':quality[0]['mean_differences']['f1']>=-.02 and gain>=.05},indent=2))
    for p,h in hashes.items():assert sha256(project/p)==h
    (out/'manifest.json').write_text(json.dumps({'hashes':hashes,'baseline':str(baseline),'model_hashes':{p.name:sha256(p) for p in (out/'models').iterdir()},'test_evaluated':False,'speaker_validated':False},indent=2))
    return out


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    with threadpool_limits(limits=1):print(run(Path(__file__).resolve().parents[2],Path(sys.argv[1])))
