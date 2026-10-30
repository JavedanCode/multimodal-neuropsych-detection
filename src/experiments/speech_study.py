"""Training-only nested evaluation of frozen Wav2Vec2 speech embeddings."""
from datetime import datetime,timezone
import json
import logging
from pathlib import Path
import subprocess
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.models.representation_preprocessing import RepresentationPreprocessor
from src.experiments.audio_study import SEEDS,select_inner,estimator,metric_row
from src.experiments.baselines import sha256,validate_partition,TARGET_POLICY


def load_inputs(project):
    root=project/'data/processed';directory=root/'speech_study_v1'
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['partition']!='train':raise ValueError('Training-only speech required')
    if manifest['contract']['source']!=sha256(project/'src/dataset/speech_embeddings.py'):raise ValueError('Extractor changed')
    for name,digest in manifest['output_hashes'].items():
        if sha256(directory/name)!=digest:raise ValueError('Speech table changed')
    for name,digest in manifest['source_tables'].items():
        if sha256(root/name)!=digest:raise ValueError('Source table changed')
    official=pd.read_parquet(root/'labels.parquet')
    y=pd.read_parquet(directory/'labels.parquet').PHQ_Binary
    previous=pd.read_parquet(root/'segmentation_study_v1/labels.parquet').PHQ_Binary
    ids=official.index[(official.split=='train')&~official.index.isin(manifest['excluded_ids'])]
    if not y.equals(official.loc[ids,'PHQ_Binary']) or not y.equals(previous):raise ValueError('Unchanged training cohort required')
    base=pd.read_parquet(root/'features.parquet').loc[ids]
    speech=pd.read_parquet(directory/'features.parquet')
    temporal=pd.read_parquet(root/'representation_study_v1/facial.parquet')
    facial_manifest=json.loads((root/'representation_study_v1/manifest.json').read_text())
    if sha256(root/'representation_study_v1/facial.parquet')!=facial_manifest['output_hashes']['facial.parquet']:raise ValueError('Facial table changed')
    if not speech.index.equals(ids) or not temporal.index.equals(ids):raise ValueError('Participant order mismatch')
    audio=base.filter(like='audio__');facial=base.filter(like='facial__')
    sets={'base_audio':audio,'base_concat':base,'speech_audio':speech,
          'speech_base_facial':pd.concat([speech,facial],axis=1),
          'speech_temporal_facial':pd.concat([speech,temporal],axis=1)}
    for x in sets.values():validate_partition(x,y)
    return sets,y,manifest


def prepare_inner(x,y,splits):
    prepared=[]
    for a,b in splits:
        prep=RepresentationPreprocessor().fit(x.iloc[a])
        prepared.append((a,b,prep.transform(x.iloc[a]),prep.transform(x.iloc[b])))
    return prepared


def fit_outer(x,y,valid,family,selected,seed):
    model=Pipeline([('preprocess',RepresentationPreprocessor()),('classifier',estimator(family,selected['parameters'],seed))])
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning);model.fit(x,y)
    scores=model.decision_function(valid)
    if not np.isfinite(scores).all():raise ValueError('Nonfinite outer scores')
    return model,scores


def summarize_results(pred,folds,output):
    rows=[];subgroups=[]
    for (condition,seed),g in pred.groupby(['condition','seed']):
        rows.append({'condition':condition,'seed':int(seed),**metric_row(g.target,g.prediction),
            'mean_fold_auc':float(folds.loc[(folds.condition==condition)&(folds.seed==seed),'roc_auc'].mean())})
        for domain,d in g.groupby(np.where(g.participant_id>=600,'AI','WoZ')):
            subgroups.append({'condition':condition,'seed':int(seed),'domain':domain,'n':len(d),
                'positives':int(d.target.sum()),**metric_row(d.target,d.prediction)})
    metrics=pd.DataFrame(rows);metrics.to_csv(output/'repeat_metrics.csv',index=False)
    pd.DataFrame(subgroups).to_csv(output/'domain_metrics.csv',index=False)
    measures=['f1','precision','recall','specificity','accuracy','balanced_accuracy','mean_fold_auc']
    summary=metrics.groupby('condition')[measures].agg(['mean',lambda s:s.std(ddof=0)])
    summary.columns=[f'{a}_{"sd" if b=="<lambda_0>" else b}' for a,b in summary.columns]
    summary.to_csv(output/'summary.csv')
    paired=[];decisions=[]
    for family in ['logistic','linear']:
        for first,second in [('speech_audio','base_audio'),('speech_base_facial','base_concat'),
                ('speech_temporal_facial','base_concat')]:
            first+='_'+family;second+='_'+family
            differences=[]
            for seed in SEEDS:
                s=metrics[metrics.seed==seed].set_index('condition')
                r={'first':first,'second':second,'seed':seed,**{m:float(s.loc[first,m]-s.loc[second,m]) for m in measures}}
                paired.append(r);differences.append(r)
            d=pd.DataFrame(differences);mean=d[measures].mean()
            # Only the three preregistered primary comparisons can clear the screening rule.
            primary=True
            passes=bool(primary and mean.f1>=.03 and mean.balanced_accuracy>=0 and mean.mean_fold_auc>=0 and mean.recall>=-.05 and (d.f1>0).sum()>=2)
            decisions.append({'candidate':first,'control':second,'primary':primary,'passes_exploratory_rule':passes,
                'f1_improved_repeats':int((d.f1>0).sum()),'mean_differences':mean.to_dict()})
    pd.DataFrame(paired).to_csv(output/'paired_differences.csv',index=False)
    (output/'screening_decisions.json').write_text(json.dumps(decisions,indent=2,allow_nan=False),encoding='utf-8')
    return summary


def run(project,output):
    sets,y,manifest=load_inputs(project)
    output.mkdir(parents=True,exist_ok=False);(output/'models').mkdir()
    (output/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    paths=['docs/speech_encoder_protocol.md','src/experiments/speech_study.py',
        'src/experiments/audio_study.py','src/experiments/baselines.py','src/experiments/metrics.py',
        'src/models/representation_preprocessing.py','src/dataset/preprocessing.py','src/dataset/speech_embeddings.py',
        'data/processed/features.parquet','data/processed/labels.parquet',
        'data/processed/speech_study_v1/features.parquet','data/processed/representation_study_v1/facial.parquet',
        'data/processed/speech_study_v1/labels.parquet','data/processed/speech_study_v1/manifest.json']
    hashes={p:sha256(project/p) for p in paths}
    (output/'protocol.json').write_text(json.dumps({'hashes':hashes,'target':TARGET_POLICY,
        'excluded_ids':manifest['excluded_ids'],'participant_ids':list(map(int,y.index)),
        'seeds':SEEDS,'outer_folds':5,'inner_folds':3,'threshold':0,'development_evaluated':False,'test_evaluated':False},indent=2),encoding='utf-8')
    predictions=[];fold_metrics=[]
    for seed in SEEDS:
        for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(next(iter(sets.values())),y)):
            target=y.iloc[a];held=y.iloc[b]
            inner=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(np.zeros(len(a)),target))
            audit={'fit_ids':list(map(int,target.index)),'heldout_ids':list(map(int,held.index)),
                'inner_ids':[{'fit':list(map(int,target.index[ia])),'heldout':list(map(int,target.index[ib]))} for ia,ib in inner],
                'candidates':{},'selected':{}}
            for name,x in sets.items():
                logging.info('Speech encoder seed %s fold %s/5: %s',seed,fold+1,name)
                fit=x.iloc[a];valid=x.iloc[b];prepared=prepare_inner(fit,target,inner)
                for family in ['logistic','linear']:
                    condition=name+'_'+family;selected,_,records=select_inner(fit,target,family,prepared,seed,[0.])
                    audit['candidates'][condition]=records;audit['selected'][condition]=selected
                    model,scores=fit_outer(fit,target,valid,family,selected,seed)
                    checkpoint=output/'models'/f'{condition}_{seed}_{fold}.joblib';joblib.dump(model,checkpoint)
                    np.testing.assert_array_equal(scores,joblib.load(checkpoint).decision_function(valid))
                    pred=scores>=0
                    fold_metrics.append({'condition':condition,'seed':seed,'fold':fold,
                        'retained_features':len(model.named_steps['preprocess'].columns_),
                        'roc_auc':float(roc_auc_score(held,scores)),**metric_row(held,pred)})
                    predictions.extend({'condition':condition,'seed':seed,'fold':fold,'participant_id':int(pid),
                        'target':int(t),'score':float(s),'prediction':int(p)} for pid,t,s,p in zip(held.index,held,scores,pred))
            for constant in [0,1]:
                condition=f'constant_{constant}';pred=np.full(len(held),constant)
                fold_metrics.append({'condition':condition,'seed':seed,'fold':fold,'roc_auc':.5,**metric_row(held,pred)})
                predictions.extend({'condition':condition,'seed':seed,'fold':fold,'participant_id':int(pid),
                    'target':int(t),'score':float(constant),'prediction':constant} for pid,t in held.items())
            (output/f'fold_{seed}_{fold}.json').write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
    pred=pd.DataFrame(predictions);folds=pd.DataFrame(fold_metrics)
    for _,g in pred.groupby(['condition','seed']):
        if len(g)!=len(y) or not g.participant_id.is_unique or set(g.participant_id)!=set(y.index):raise ValueError('OOF coverage mismatch')
    pred.to_parquet(output/'predictions.parquet',index=False);folds.to_csv(output/'fold_metrics.csv',index=False)
    summary=summarize_results(pred,folds,output)
    for p,h in hashes.items():
        if sha256(project/p)!=h:raise ValueError('Source changed during fitting')
    checkpoints={p.name:sha256(p) for p in (output/'models').glob('*.joblib')}
    (output/'verification.json').write_text(json.dumps({'participants':len(y),'positives':int(y.sum()),
        'restored_models':len(checkpoints),'checkpoint_hashes':checkpoints,'prediction_rows':len(pred),
        'input_hashes_verified':True,'test_evaluated':False,'development_evaluated':False},indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    project=Path(__file__).resolve().parents[2]
    output=project/'experiments/runs'/('speech_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    with threadpool_limits(limits=1):summary=run(project,output)
    print(summary.to_string());print(output)
