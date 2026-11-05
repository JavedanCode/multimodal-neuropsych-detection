"""Sequential precision-focused experiments on fixed training folds."""
from datetime import datetime, timezone
from pathlib import Path
import json
import logging
import sys
import subprocess
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from src.experiments.audio_study import SEEDS, estimator, candidates
from src.experiments.baselines import sha256
from src.experiments.mpnet_study import inputs
from src.experiments.text_temporal_study import summarize, save_predictions
from src.models.text_preprocessing import TextPreprocessor

EXPERIMENTS = ['threshold', 'cost', 'rbf', 'agreement']


def measures(y, scores):
    p=np.asarray(scores)>=0
    return {'precision':float(precision_score(y,p,zero_division=0)),
            'recall':float(recall_score(y,p,zero_division=0)),
            'f1':float(f1_score(y,p,zero_division=0))}


def choose(records, reference):
    eligible=[r for r in records if r['metrics']['recall']>=.65 and r['metrics']['f1']>=reference['metrics']['f1']-.02]
    return max(eligible,key=lambda r:(r['metrics']['precision'],r['metrics']['f1'],r['metrics']['recall'])) if eligible else reference


def decision(bundle, x):
    z=bundle['preprocess'].transform(x)
    raw=[m.decision_function(z) for m in bundle['models']]
    return np.min(raw,axis=0)-bundle['threshold']


def inner_scores(prepared,target,family,params,seed):
    scores=np.full(len(target),np.nan)
    for ia,ib,z,v in prepared:
        scores[ib]=estimator(family,params,seed).fit(z,target.iloc[ia]).decision_function(v)
    if not np.isfinite(scores).all():raise ValueError('Incomplete inner scores')
    return scores


def record(kind,params,threshold,scores,target):
    return {'kind':kind,'parameters':params,'threshold':threshold,
            'scores':scores.tolist(),'metrics':measures(target,scores)}


def run(project,experiment):
    if experiment not in EXPERIMENTS:raise ValueError('Unknown experiment')
    sets,y=inputs(project);x=sets['mpnet']
    out=project/'experiments/runs'/('precision_'+experiment+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir();(out/'models').mkdir()
    paths=['docs/precision_protocol.md','src/experiments/precision_series.py',
           'src/models/text_preprocessing.py','src/experiments/audio_study.py',
           'src/experiments/mpnet_study.py','src/experiments/text_temporal_study.py',
           'src/experiments/metrics.py','data/processed/mpnet_v1/features.parquet',
           'data/processed/mpnet_v1/labels.parquet','data/processed/mpnet_v1/manifest.json']
    hashes={p:sha256(project/p) for p in paths}
    (out/'protocol.json').write_text(json.dumps({'experiment':experiment,'hashes':hashes,
        'participants':list(map(int,y.index)),'development_evaluated':False,'test_evaluated':False},indent=2))
    (out/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    predictions=[]
    for seed in SEEDS:
        for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(x,y)):
            logging.info('%s seed %s fold %s',experiment,seed,fold)
            fit,target,valid,held=x.iloc[a],y.iloc[a],x.iloc[b],y.iloc[b]
            splits=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(fit,target))
            prepared=[]
            for ia,ib in splits:
                prep=TextPreprocessor().fit(fit.iloc[ia])
                prepared.append((ia,ib,prep.transform(fit.iloc[ia]),prep.transform(fit.iloc[ib])))
            references=[]
            for params in candidates('linear'):
                scores=inner_scores(prepared,target,'linear',params,seed)
                references.append(record('reference',params,0.,scores,target))
            reference=max(references,key=lambda r:r['metrics']['f1'])
            records=[reference];base=np.array(reference['scores'])
            if experiment=='threshold':
                for cutoff in [.1,.2,.3,.5,.75,1.]:
                    records.append(record('threshold',reference['parameters'],cutoff,base-cutoff,target))
            elif experiment=='cost':
                for c in [.01,.1,1.]:
                    for weight in [.75,1.,1.5,2.,3.]:
                        params={'C':c,'class_weight':{0:1.,1:weight}}
                        records.append(record('cost',params,0.,inner_scores(prepared,target,'linear',params,seed),target))
            elif experiment=='rbf':
                for c in [.1,1.,10.]:
                    for gamma in ['scale',.01,.1]:
                        for weight in [None,'balanced']:
                            params={'C':c,'gamma':gamma,'class_weight':weight}
                            records.append(record('rbf',params,0.,inner_scores(prepared,target,'rbf',params,seed),target))
            else:
                for params in candidates('logistic'):
                    scores=np.minimum(base,inner_scores(prepared,target,'logistic',params,seed))
                    records.append(record('agreement',params,0.,scores,target))
            selected=choose(records,reference)
            prep=TextPreprocessor().fit(fit);z=prep.transform(fit)
            for name,spec in [('reference',reference),(experiment,selected)]:
                family='rbf' if spec['kind']=='rbf' else ('logistic' if spec['kind']=='agreement' else 'linear')
                models=[estimator(family,spec['parameters'],seed).fit(z,target)]
                if spec['kind']=='agreement':models.insert(0,estimator('linear',reference['parameters'],seed).fit(z,target))
                bundle={'preprocess':prep,'models':models,'threshold':spec['threshold']}
                joblib.dump(bundle,out/'models'/f'{name}_{seed}_{fold}.joblib')
                save_predictions(predictions,name,'clean',seed,fold,held,decision(bundle,valid))
            audit={'fit_ids':list(map(int,target.index)),'heldout_ids':list(map(int,held.index)),
                   'inner_ids':[{'fit':list(map(int,target.index[ia])),'heldout':list(map(int,target.index[ib]))} for ia,ib in splits],
                   'reference_candidates':references,'reference':reference,'candidates':records,'selected':selected}
            (out/f'fold_{seed}_{fold}.json').write_text(json.dumps(audit,indent=2,allow_nan=False))
    pred=pd.DataFrame(predictions);pred.to_parquet(out/'predictions.parquet',index=False);metrics=summarize(pred,out)
    dif=metrics[metrics.condition==experiment].set_index('seed').select_dtypes('number')-metrics[metrics.condition=='reference'].set_index('seed').select_dtypes('number')
    mean=dif.mean();actual=metrics[metrics.condition==experiment].recall.mean()
    checks={'precision_gain_5pp':bool(mean.precision>=.05),'recall_loss_at_most_5pp':bool(mean.recall>=-.05),
            'recall_at_least_65':bool(actual>=.65),'f1_loss_at_most_2pp':bool(mean.f1>=-.02),
            'precision_gain_two_repeats':bool((dif.precision>0).sum()>=2)}
    (out/'screening_decision.json').write_text(json.dumps({'checks':checks,'passes':all(checks.values()),'mean_differences':mean.to_dict()},indent=2))
    for p,h in hashes.items():assert sha256(project/p)==h
    (out/'verification.json').write_text(json.dumps({'model_hashes':{p.name:sha256(p) for p in (out/'models').iterdir()},'test_evaluated':False},indent=2))
    return out


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    with threadpool_limits(limits=1):print(run(Path(__file__).resolve().parents[2],sys.argv[1]))
