"""Matched nested transcript-coverage/facial-fusion pilot; training only."""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.experiments.audio_study import SEEDS, inner_data, select_inner, fit_outer, metric_row
from src.experiments.baselines import sha256, validate_partition, TARGET_POLICY


def load_inputs(project):
    root=project/'data/processed';directory=root/'segmentation_study_v1'
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['partition']!='train':raise ValueError('Training-only tables required')
    for name,digest in manifest['output_hashes'].items():
        if sha256(directory/name)!=digest:raise ValueError('Table hash mismatch')
    for name,digest in manifest['sources'].items():
        if sha256(project/name)!=digest:raise ValueError('Extractor source changed')
    covered=pd.read_parquet(directory/'features.parquet')
    y=pd.read_parquet(directory/'labels.parquet').PHQ_Binary
    official=pd.read_parquet(root/'labels.parquet')
    expected=official.index[(official.split=='train')&~official.index.isin(manifest['excluded_ids'])]
    if not covered.index.equals(expected) or not y.equals(official.loc[expected,'PHQ_Binary']):
        raise ValueError('Unchanged training subset required')
    for pid in manifest['excluded_ids']:
        if official.loc[pid,'split']!='train':raise ValueError('Invalid quarantine')
        audit=json.loads((project/'reports/data_audit/label_speech_cache'/f'{pid}.json').read_text())
        if abs(audit['result']['end_gap_seconds'])<=1:raise ValueError('Quarantine lacks timing evidence')
    full=pd.read_parquet(root/'features.parquet').loc[expected]
    facial=full.filter(like='facial__');audio=full.filter(like='audio__')
    if list(covered)!=list(audio):raise ValueError('Audio schema mismatch')
    sets={'full_audio':audio,'covered_audio':covered,'facial_only':facial,
          'full_audio_facial':pd.concat([audio,facial],axis=1),
          'covered_audio_facial':pd.concat([covered,facial],axis=1)}
    for x in sets.values():validate_partition(x,y)
    return sets,y,manifest


def run(project,output):
    sets,y,manifest=load_inputs(project)
    output.mkdir(parents=True,exist_ok=False);(output/'models').mkdir()
    (output/'environment.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    paths=['docs/segmentation_study_protocol.md','src/experiments/segmentation_study.py',
           'src/experiments/audio_study.py','src/experiments/baselines.py','src/experiments/metrics.py',
           'src/dataset/preprocessing.py','src/dataset/segmented_audio.py','src/dataset/label_speech_audit.py',
           'data/processed/features.parquet','data/processed/labels.parquet',
           'data/processed/segmentation_study_v1/features.parquet',
           'data/processed/segmentation_study_v1/labels.parquet','data/processed/segmentation_study_v1/manifest.json']
    provenance={p:sha256(project/p) for p in paths}
    (output/'protocol.json').write_text(json.dumps({'hashes':provenance,'target':TARGET_POLICY,
        'excluded_ids':manifest['excluded_ids'],'participant_ids':list(map(int,y.index)),
        'seeds':SEEDS,'outer_folds':5,'inner_folds':3,'cutoff':0,
        'development_evaluated':False,'test_evaluated':False},indent=2),encoding='utf-8')
    predictions=[];fold_metrics=[]
    for seed in SEEDS:
        for fold,(a,b) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(next(iter(sets.values())),y)):
            logging.info('Coverage/fusion study: seed %s fold %s/5',seed,fold+1)
            target=y.iloc[a];held=y.iloc[b]
            inner=list(StratifiedKFold(3,shuffle=True,random_state=seed+fold+1).split(np.zeros(len(a)),target))
            audit={'fit_ids':list(map(int,target.index)),'heldout_ids':list(map(int,held.index)),
                   'inner_ids':[{'fit':list(map(int,target.index[ia])),'heldout':list(map(int,target.index[ib]))} for ia,ib in inner],
                   'candidates':{},'selected':{}}
            for name,x in sets.items():
                fit=x.iloc[a];valid=x.iloc[b];prepared=inner_data(fit,target,inner)
                for family in ['logistic','linear']:
                    condition=name+'_'+family
                    selected,_,records=select_inner(fit,target,family,prepared,seed,[0.])
                    audit['candidates'][condition]=records;audit['selected'][condition]=selected
                    model,scores=fit_outer(fit,target,valid,family,selected,seed)
                    checkpoint=output/'models'/f'{condition}_{seed}_{fold}.joblib'
                    joblib.dump(model,checkpoint)
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
    pred.to_parquet(output/'predictions.parquet',index=False);folds.to_csv(output/'fold_metrics.csv',index=False)
    rows=[];subgroups=[]
    for (condition,seed),g in pred.groupby(['condition','seed']):
        if len(g)!=len(y) or set(g.participant_id)!=set(y.index):raise ValueError('OOF coverage mismatch')
        rows.append({'condition':condition,'seed':seed,**metric_row(g.target,g.prediction),
                     'mean_fold_auc':folds.loc[(folds.condition==condition)&(folds.seed==seed),'roc_auc'].mean()})
        for domain,d in g.groupby(np.where(g.participant_id>=600,'AI','WoZ')):
            subgroups.append({'condition':condition,'seed':seed,'domain':domain,'n':len(d),
                'positives':int(d.target.sum()),**metric_row(d.target,d.prediction)})
    metrics=pd.DataFrame(rows);metrics.to_csv(output/'repeat_metrics.csv',index=False)
    pd.DataFrame(subgroups).to_csv(output/'domain_metrics.csv',index=False)
    measures=['f1','precision','recall','specificity','accuracy','balanced_accuracy','mean_fold_auc']
    summary=metrics.groupby('condition')[measures].agg(['mean',lambda s:s.std(ddof=0)])
    summary.columns=[f'{a}_{"sd" if b=="<lambda_0>" else b}' for a,b in summary.columns]
    summary.to_csv(output/'summary.csv')
    paired=[]
    for seed in SEEDS:
        scores=metrics[metrics.seed==seed].set_index('condition')
        for family in ['logistic','linear']:
            for first,second,effect in [('covered_audio','full_audio','coverage'),
                ('covered_audio_facial','full_audio_facial','coverage_with_facial'),
                ('full_audio_facial','full_audio','add_facial_full'),
                ('covered_audio_facial','covered_audio','add_facial_covered')]:
                paired.append({'seed':seed,'family':family,'effect':effect,**{
                    m:float(scores.loc[first+'_'+family,m]-scores.loc[second+'_'+family,m]) for m in measures}})
    pd.DataFrame(paired).to_csv(output/'paired_differences.csv',index=False)
    for name,digest in provenance.items():
        if sha256(project/name)!=digest:raise ValueError('Input changed during run')
    (output/'verification.json').write_text(json.dumps({'participants':len(y),'positives':int(y.sum()),
        'restored_models':len(list((output/'models').glob('*.joblib'))),'prediction_rows':len(pred),'input_hashes_verified':True,
        'checkpoint_hashes':{p.name:sha256(p) for p in (output/'models').glob('*.joblib')},
        'test_evaluated':False,'development_evaluated':False},indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    project=Path(__file__).resolve().parents[2]
    output=project/'experiments/runs'/('segmentation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    print(run(project,output).to_string());print(output)
