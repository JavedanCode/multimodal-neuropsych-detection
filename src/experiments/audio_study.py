"""Training-only nested audio study; no development or test prediction API."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.dataset.preprocessing import ParticipantPreprocessor
from src.experiments.baselines import sha256, validate_partition, TARGET_POLICY
from src.experiments.metrics import binary_metrics

SEEDS=[42,137,2026]
THRESHOLDS=[0.,-.5,.5,-1.,1.]


def candidates(family):
    return [{'C':c,'class_weight':weight,**({'gamma':gamma} if family=='rbf' else {})}
            for c in [.01,.1,1.] for weight in [None,'balanced']
            for gamma in (['scale',.01] if family=='rbf' else [None])]


def estimator(family, params, seed):
    if family=='logistic':
        return LogisticRegression(solver='liblinear',max_iter=5000,random_state=seed,**params)
    if family not in ['linear','rbf']:raise ValueError('Unknown model family')
    # Default is uncalibrated margins. Explicit probability=False is deprecated
    # by sklearn 1.9; omitting it also preserves older versions' false default.
    return SVC(kernel=family,cache_size=64,random_state=seed,**params)


def inner_data(x,y,splits):
    """Fit preprocessing only on each inner fitting fold; reuse across candidates."""
    result=[]
    for a,b in splits:
        prep=ParticipantPreprocessor().fit(x.iloc[a])
        result.append((a,b,prep.transform(x.iloc[a]),prep.transform(x.iloc[b])))
    return result


def select_inner(x,y,family,prepared,seed,thresholds=THRESHOLDS,mean_fold=False):
    """Only inner data/labels are accepted; outer outcomes cannot select settings."""
    records=[]
    for params in candidates(family):
        oof=np.full(len(y),np.nan)
        for a,b,fit,valid in prepared:
            model=estimator(family,params,seed)
            with warnings.catch_warnings():
                warnings.simplefilter('error',ConvergenceWarning)
                model.fit(fit,y.iloc[a])
            oof[b]=model.decision_function(valid)
        if not np.isfinite(oof).all():raise ValueError('Incomplete/nonfinite inner OOF scores')
        for threshold in thresholds:
            predicted=oof>=threshold
            fold_scores=[float(f1_score(y.iloc[b],predicted[b],zero_division=0)) for _,b,_,_ in prepared]
            pooled=float(f1_score(y,predicted,zero_division=0))
            records.append({'parameters':params,'threshold':threshold,'pooled_f1':pooled,
                'fold_f1':fold_scores,'selection_f1':float(np.mean(fold_scores)) if mean_fold else pooled})
    # Python max returns the first tied item, preserving the predeclared order.
    default=max([r for r in records if r['threshold']==0],key=lambda r:r['selection_f1'])
    tuned=max(records,key=lambda r:r['selection_f1'])
    return default,tuned,records


def fit_outer(x,y,valid,family,selected,seed):
    model=Pipeline([('preprocess',ParticipantPreprocessor()),
                    ('classifier',estimator(family,selected['parameters'],seed))])
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        model.fit(x,y)
    scores=model.decision_function(valid)
    if not np.isfinite(scores).all():raise ValueError('Nonfinite outer scores')
    return model,scores


def metric_row(y,predicted):
    metrics=binary_metrics(y,np.asarray(predicted,dtype=float))
    result={k:metrics[k] for k in ['precision','recall','f1','macro_f1','accuracy','specificity']}
    result['balanced_accuracy']=(result['recall']+result['specificity'])/2
    result.update(zip(['tn','fp','fn','tp'],np.array(metrics['confusion_matrix']).ravel().tolist()))
    return result


def load_inputs(project):
    directory=project/'data/processed/audio_study_v1'
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['partition']!='train':raise ValueError('Training-only enriched summaries required')
    for name,digest in manifest['output_sha256'].items():
        if sha256(directory/name)!=digest:raise ValueError('Enriched table provenance mismatch')
    if manifest['contract']['source']!=sha256(project/'src/dataset/audio_summary.py'):
        raise ValueError('Extractor changed; rebuild summaries')
    x=pd.read_parquet(directory/'features.parquet');y=pd.read_parquet(directory/'labels.parquet').PHQ_Binary
    official=pd.read_parquet(project/'data/processed/labels.parquet')
    train_ids=official.index[official.split=='train']
    if not x.index.equals(train_ids) or not y.equals(official.loc[train_ids,'PHQ_Binary']):
        raise ValueError('Study must contain unchanged official training participants/targets only')
    validate_partition(x,y)
    return x,y,manifest['baseline_columns']


def run(project,output):
    x,y,base=load_inputs(project)
    output.mkdir(parents=True,exist_ok=False)
    (output/'models').mkdir()
    inputs=project/'data/processed/audio_study_v1'
    hashes={str(p.relative_to(project)):sha256(p) for p in [inputs/'features.parquet',inputs/'labels.parquet',inputs/'manifest.json',
        project/'src/experiments/audio_study.py',project/'src/dataset/audio_summary.py',
        project/'src/dataset/preprocessing.py',project/'docs/audio_study_protocol.md']}
    (output/'protocol.json').write_text(json.dumps({'target_policy':TARGET_POLICY,'seeds':SEEDS,
        'outer_folds':5,'inner_folds':3,'baseline_inner_folds':5,'thresholds':THRESHOLDS,
        'input_source_sha256':hashes,'test_evaluated':False,'development_evaluated':False},indent=2),encoding='utf-8')
    predictions=[];fold_results=[];fold_audit=[]
    started=time.perf_counter()
    for seed in SEEDS:
        outer=list(StratifiedKFold(5,shuffle=True,random_state=seed).split(x,y))
        for fold,(a,b) in enumerate(outer):
            logging.info('Nested audio study: seed %s, outer fold %s/5',seed,fold+1)
            fit,target,valid=x.iloc[a],y.iloc[a],x.iloc[b]
            inner=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(fit,target))
            audit={'seed':seed,'fold':fold,'fit_ids':list(map(int,fit.index)),
                'heldout_ids':list(map(int,valid.index)),'inner_participants':[{
                    'fit':list(map(int,fit.index[ia])),'validation':list(map(int,fit.index[ib]))} for ia,ib in inner],
                'candidates':{}}

            def save(condition,family,columns,selected):
                model,scores=fit_outer(fit[columns],target,valid[columns],family,selected,seed)
                pred=scores>=selected['threshold']
                name=f'{condition}_seed{seed}_fold{fold}'
                joblib.dump(model,output/'models'/f'{name}.joblib')
                # Verify serialization before using the predictions in results.
                restored=joblib.load(output/'models'/f'{name}.joblib').decision_function(valid[columns])
                np.testing.assert_array_equal(scores,restored)
                fold_results.append({'condition':condition,'seed':seed,'fold':fold,
                    'threshold':selected['threshold'],'C':selected['parameters']['C'],
                    'class_weight':selected['parameters']['class_weight'],
                    'gamma':selected['parameters'].get('gamma'),
                    'retained_features':len(model.named_steps['preprocess'].columns_),
                    'roc_auc':float(roc_auc_score(y.iloc[b],scores)),**metric_row(y.iloc[b],pred)})
                predictions.extend({'condition':condition,'seed':seed,'fold':fold,'participant_id':int(pid),
                    'PHQ_Binary':int(t),'decision_score':float(score),'threshold':selected['threshold'],
                    'prediction':int(p)} for pid,t,score,p in zip(valid.index,y.iloc[b],scores,pred))

            for feature_set,columns in [('base',base),('rich',list(x))]:
                prepared=inner_data(fit[columns],target,inner)
                for family in ['logistic','linear','rbf']:
                    default,tuned,records=select_inner(fit[columns],target,family,prepared,seed)
                    audit['candidates'][feature_set+'_'+family]=records
                    for method,selected in [('default',default),('tuned',tuned)]:
                        save(feature_set+'_'+family+'_'+method,family,columns,selected)
            baseline_inner=list(StratifiedKFold(5,shuffle=True,random_state=seed+fold+1).split(fit,target))
            prepared=inner_data(fit[base],target,baseline_inner)
            selected,_,records=select_inner(fit[base],target,'logistic',prepared,seed,[0.],mean_fold=True)
            audit['baseline_inner_participants']=[{'fit':list(map(int,fit.index[ia])),
                'validation':list(map(int,fit.index[ib]))} for ia,ib in baseline_inner]
            audit['candidates']['stage4_recipe']=records
            save('stage4_recipe','logistic',base,selected)
            fold_audit.append(audit)
            (output/f'fold_seed{seed}_{fold}.json').write_text(json.dumps(audit,indent=2,allow_nan=False),encoding='utf-8')
    pred=pd.DataFrame(predictions);folds=pd.DataFrame(fold_results)
    pred.to_parquet(output/'outer_predictions.parquet',index=False)
    folds.to_csv(output/'outer_fold_metrics.csv',index=False)
    rows=[]
    for (condition,seed),group in pred.groupby(['condition','seed']):
        if len(group)!=len(y) or set(group.participant_id)!=set(y.index):raise ValueError('Outer coverage mismatch')
        row={'condition':condition,'seed':int(seed),**metric_row(group.PHQ_Binary,group.prediction)}
        row['mean_fold_auc']=float(folds[(folds.condition==condition)&(folds.seed==seed)].roc_auc.mean())
        rows.append(row)
    metrics=pd.DataFrame(rows);metrics.to_csv(output/'repeat_metrics.csv',index=False)
    summary=[]
    for condition,group in metrics.groupby('condition'):
        item={'condition':condition}
        for metric in ['f1','precision','recall','specificity','accuracy','balanced_accuracy','mean_fold_auc']:
            item[metric+'_mean']=float(group[metric].mean());item[metric+'_sd']=float(group[metric].std(ddof=0))
        summary.append(item)
    summary=pd.DataFrame(summary);summary.to_csv(output/'summary.csv',index=False)
    comparisons=[]
    for seed in SEEDS:
        values=metrics[metrics.seed==seed].set_index('condition').f1
        for condition,value in values.items():
            comparisons.append({'seed':seed,'comparison':'vs_stage4','first':condition,'second':'stage4_recipe',
                                'f1_difference':float(value-values['stage4_recipe'])})
        for family in ['logistic','linear','rbf']:
            pairs=[(f'rich_{family}_{method}',f'base_{family}_{method}','richer_features') for method in ['default','tuned']]
            pairs += [(f'{features}_{family}_tuned',f'{features}_{family}_default','tuned_threshold') for features in ['base','rich']]
            for first,second,effect in pairs:
                comparisons.append({'seed':seed,'comparison':effect,'first':first,'second':second,
                                    'f1_difference':float(values[first]-values[second])})
    pd.DataFrame(comparisons).to_csv(output/'paired_repeat_differences.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained')
    ax.barh(summary.condition,summary.f1_mean,xerr=summary.f1_sd,capsize=3)
    ax.invert_yaxis();ax.set(xlim=(0,1),xlabel='Outer-held-out F1: repeat mean and SD (not CI)',
                           title='Exploratory audio study: official training participants only')
    for suffix in ['png','svg']:fig.savefig(output/f'nested_f1.{suffix}',dpi=180)
    plt.close(fig)
    lines=['# Follow-up audio study: nested training-only results','',
        'All 163 official training participants; 5 outer folds x 3 repeats. No development/test predictions.',
        'Released labels remain provisional. Repeats share participants; SD is not a CI.','',
        '| Condition | F1 mean +/- SD | Precision | Recall | Mean fold AUC |',
        '|---|---:|---:|---:|---:|']
    for _,r in summary.iterrows():
        lines.append(f'| {r.condition} | {r.f1_mean:.3f} +/- {r.f1_sd:.3f} | {r.precision_mean:.3f} | {r.recall_mean:.3f} | {r.mean_fold_auc_mean:.3f} |')
    lines += ['','Every threshold and hyperparameter choice used inner folds only.',
        'SVM decision scores are margins, not probabilities. AUC is averaged by fold to avoid pooling incompatible score scales.',
        'All methods use identical outer folds. The Stage 4 recipe is rerun inside this protocol for a matched comparison.',
        'No independent confirmation or clinical-validity claim. Do not compare these CV scores directly with prior test F1.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    report={'target_policy':TARGET_POLICY,'test_evaluated':False,'development_evaluated':False,
        'participants':len(y),'conditions':13,'repeats':3,'outer_models_verified':len(folds),
        'elapsed_seconds':time.perf_counter()-started,'input_source_sha256':hashes,
        'checkpoint_sha256':{p.name:sha256(p) for p in (output/'models').glob('*.joblib')}}
    (output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    project=Path(__file__).resolve().parents[2]
    output=project/'experiments/runs'/('audio_study_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    run(project,output)
    print(output)
