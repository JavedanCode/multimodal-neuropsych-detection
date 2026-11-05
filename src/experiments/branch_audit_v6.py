"""Same-fold primary representation audit followed by matched fusion controls."""
from pathlib import Path
from datetime import datetime,timezone
import json
import logging
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,average_precision_score
from threadpoolctl import threadpool_limits
from src.experiments.baselines import sha256
from src.experiments.audio_study import load_inputs as load_audio,estimator,SEEDS
from src.experiments.representation_study import load_inputs as load_representations
from src.experiments.aligned_fusion_v2 import load as load_aligned
from src.experiments.domain_v2 import splits
from src.experiments.text_temporal_study import save_predictions,summarize
from src.experiments.compare_text_temporal import paired
from src.models.branch_audit_v6 import BranchPreprocessor,fit_network,neural_scores

NAMES=('original_audio','aligned_audio','rich_audio','original_face','aligned_face','rich_face')


def load(project):
    aligned,y,_=load_aligned(project);old,oy,_=load_representations(project);rich,ry,_=load_audio(project)
    pd.testing.assert_series_equal(y,oy);pd.testing.assert_series_equal(y,ry.loc[y.index])
    sets={'original_audio':old['base_audio'].copy(),
          'aligned_audio':pd.DataFrame(aligned['audio'],index=y.index,columns=[f'audio__aligned_{i}' for i in range(aligned['audio'].shape[1])]),
          'rich_audio':rich.loc[y.index].copy(),'original_face':old['base_facial'].copy(),
          'aligned_face':pd.DataFrame(aligned['face'],index=y.index,columns=[f'facial__aligned_{i}' for i in range(aligned['face'].shape[1])]),
          'rich_face':old['temporal_facial'].copy()}
    for name,x in sets.items():
        assert x.index.equals(y.index)
        if name.endswith('face'):x.loc[~aligned['available'][:,2].astype(bool)]=np.nan
    return sets,y,aligned['quality'][:,:9],aligned['available'][:,1:]


def inner_candidates(sets,y,seed,fold):
    inner=splits(y,3,seed+fold+1);records={};selected={}
    for name in NAMES:
        x=sets[name];prepared=[]
        for a,b in inner:
            prep=BranchPreprocessor().fit(x.iloc[a]);prepared.append((a,b,prep.transform(x.iloc[a]),prep.transform(x.iloc[b])))
        candidates=[]
        for family in ['logistic','linear']:
            for c in [.01,.1,1.]:
                scores=np.full(len(y),np.nan)
                for a,b,z,v in prepared:scores[b]=estimator(family,{'C':c,'class_weight':'balanced'},seed).fit(z,y.iloc[a]).decision_function(v)
                candidates.append({'feature':name,'family':family,'C':c,'scores':scores.tolist(),'inner_f1':float(f1_score(y,scores>=0))})
        records[name]=candidates;selected[name]=max(candidates,key=lambda r:r['inner_f1'])
    selected['selected_audio']=max([selected[n] for n in NAMES[:3]],key=lambda r:r['inner_f1'])
    selected['selected_face']=max([selected[n] for n in NAMES[3:]],key=lambda r:r['inner_f1'])
    return selected,{'candidates':records,'selected':selected,
        'inner_ids':[{'fit':list(map(int,y.index[a])),'heldout':list(map(int,y.index[b]))} for a,b in inner]}


def fit_classical(x,y,spec,seed):
    return Pipeline([('prep',BranchPreprocessor()),('model',estimator(spec['family'],{'C':spec['C'],'class_weight':'balanced'},seed))]).fit(x,y)


def predict(bundle,sets,q,available):
    if bundle['kind']=='branch':return bundle['model'].decision_function(sets[bundle['features'][0]])
    a,f=[sets[n] for n in bundle['features']]
    if bundle['kind']=='concat':return bundle['model'].decision_function(pd.concat([a,f],axis=1))
    return neural_scores(bundle,bundle['audio_prep'].transform(a),bundle['face_prep'].transform(f),bundle['quality_prep'].transform(q),available)


def run(project):
    sets,y,q,available=load(project)
    paths=['docs/branch_audit_v6_protocol.md','src/models/branch_audit_v6.py','src/models/adaptive.py','src/experiments/branch_audit_v6.py',
           'src/experiments/domain_v2.py','src/experiments/audio_study.py','src/experiments/representation_study.py',
           'src/experiments/aligned_fusion_v2.py','src/experiments/text_temporal_study.py','src/experiments/compare_text_temporal.py',
           'data/processed/features.parquet','data/processed/labels.parquet','data/processed/representation_study_v1/manifest.json',
           'data/processed/audio_study_v1/manifest.json','data/processed/aligned_v2/manifest.json']
    hashes={p:sha256(project/p) for p in paths}
    out=project/'experiments/runs'/('branch_audit_v6_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir();(out/'models').mkdir()
    (out/'protocol.json').write_text(json.dumps({'hashes':hashes,'test_evaluated':False,'feature_counts':{n:len(x.columns) for n,x in sets.items()}},indent=2))
    predictions=[];ap=[];timings=[]
    for seed in SEEDS:
        for fold,(a,b) in enumerate(splits(y,5,seed)):
            logging.info('Branch audit seed%s fold%s',seed,fold)
            target,held=y.iloc[a],y.iloc[b];train={n:x.iloc[a] for n,x in sets.items()};valid={n:x.iloc[b] for n,x in sets.items()}
            started=time.perf_counter();selected,audit=inner_candidates(train,target,seed,fold);selection_seconds=time.perf_counter()-started
            bundles={}
            for name,spec in selected.items():
                started=time.perf_counter();model=fit_classical(train[spec['feature']],target,spec,seed)
                bundles[name]={'kind':'branch','features':[spec['feature']],'model':model}
                timings.append({'condition':name,'seed':seed,'fold':fold,'fit_seconds':time.perf_counter()-started,'retained_features':len(model['prep'].keep_)})
            chosen=[selected['selected_audio']['feature'],selected['selected_face']['feature']]
            for name,features in [('original_concat',['original_audio','original_face']),('selected_concat',chosen)]:
                started=time.perf_counter();model=fit_classical(pd.concat([train[n] for n in features],axis=1),target,{'family':'logistic','C':.1},seed)
                bundles[name]={'kind':'concat','features':features,'model':model}
                timings.append({'condition':name,'seed':seed,'fold':fold,'fit_seconds':time.perf_counter()-started,'retained_features':len(model['prep'].keep_)})
            pa=BranchPreprocessor().fit(train[chosen[0]]);pf=BranchPreprocessor().fit(train[chosen[1]]);pq=StandardScaler().fit(q[a])
            za,zf,zq=pa.transform(train[chosen[0]]),pf.transform(train[chosen[1]]),pq.transform(q[a]);histories={}
            for mode in ['equal','adaptive','quality']:
                started=time.perf_counter();state,history=fit_network(za,zf,zq,available[a],target,mode,seed)
                name='neural_'+mode;bundles[name]={'kind':'neural','mode':mode,'features':chosen,'audio_prep':pa,'face_prep':pf,'quality_prep':pq,'state':state}
                histories[name]=history;timings.append({'condition':name,'seed':seed,'fold':fold,'fit_seconds':time.perf_counter()-started,'retained_features':za.shape[1]+zf.shape[1]})
            for name,bundle in bundles.items():
                joblib.dump(bundle,out/'models'/f'{name}_{seed}_{fold}.joblib')
                started=time.perf_counter();scores=predict(bundle,valid,q[b],available[b])
                next(r for r in reversed(timings) if r['condition']==name and r['seed']==seed and r['fold']==fold)['inference_seconds']=time.perf_counter()-started
                save_predictions(predictions,name,'clean',seed,fold,held,scores)
                ap.append({'condition':name,'seed':seed,'fold':fold,'average_precision':average_precision_score(held,scores)})
            for name,value in [('always_negative',-1.),('always_positive',1.)]:
                scores=np.full(len(held),value);save_predictions(predictions,name,'clean',seed,fold,held,scores)
                ap.append({'condition':name,'seed':seed,'fold':fold,'average_precision':average_precision_score(held,scores)})
            audit.update(fit_ids=list(map(int,target.index)),heldout_ids=list(map(int,held.index)),selection_seconds=selection_seconds,learning_curves=histories)
            (out/f'fold_{seed}_{fold}.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
    pred=pd.DataFrame(predictions);pred.to_parquet(out/'predictions.parquet',index=False);metrics=summarize(pred,out)
    pd.DataFrame(ap).to_csv(out/'fold_average_precision.csv',index=False);pd.DataFrame(timings).to_csv(out/'timings.csv',index=False)
    pairs=[(f'{r}_{m}',f'original_{m}') for m in ['audio','face'] for r in ['aligned','rich','selected']]
    pairs += [('selected_concat','original_concat'),('neural_adaptive','neural_equal'),('neural_quality','neural_adaptive')]
    pairs += [(f'neural_{mode}','original_concat') for mode in ['equal','adaptive','quality']]
    (out/'comparison.json').write_text(json.dumps([paired(metrics,a,b) for a,b in pairs],indent=2))
    for p,h in hashes.items():assert sha256(project/p)==h
    (out/'manifest.json').write_text(json.dumps({'hashes':hashes,'models':{p.name:sha256(p) for p in (out/'models').iterdir()},
        'development_evaluated':False,'test_evaluated':False,'text_predictors':False},indent=2));return out


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    with threadpool_limits(limits=1):print(run(Path(__file__).resolve().parents[2]))
