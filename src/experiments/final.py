"""Stage 9: prepare frozen ablations, then explicitly evaluate held-out data."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch

from src.experiments.adaptive import predict_checkpoint, seed_everything
from src.experiments.baselines import load_development_inputs, sha256, TARGET_POLICY, validate_partition
from src.experiments.metrics import binary_metrics
from src.experiments.selection import fit_selected

SEEDS = [42,137,2026]
METRICS = ['precision','recall','f1','macro_f1','roc_auc','accuracy','specificity']


def write_json(path, value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def bootstrap_indices(y, repeats=2000, seed=91042):
    y = np.asarray(y)
    if y.ndim != 1 or set(np.unique(y)) != {0,1} or repeats < 1:
        raise ValueError('Both binary classes and positive repeats required')
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(np.flatnonzero(y==c),size=(repeats,int((y==c).sum())),replace=True)
                           for c in [0,1]],axis=1)


def bootstrap_f1(y, p, indices):
    y, predicted = np.asarray(y)[indices], np.asarray(p)[indices] >= .5
    tp = ((y==1)&predicted).sum(1)
    denominator = 2*tp + ((y==0)&predicted).sum(1) + ((y==1)&~predicted).sum(1)
    return np.divide(2*tp,denominator,out=np.zeros(len(tp),dtype=float),where=denominator!=0)


def interval(values):
    return [float(x) for x in np.percentile(values,[2.5,97.5])]


def source_hashes(project):
    paths = sorted((project/'src').rglob('*.py')) + [project/'docs/stage9_protocol.md']
    return {str(p.relative_to(project)):sha256(p) for p in paths}


def verify_manifest(manifest, project):
    if manifest['target_policy'] != TARGET_POLICY or manifest['threshold'] != .5:
        raise ValueError('Unsupported final target/threshold')
    for relative, digest in manifest['frozen_hashes'].items():
        if sha256(project/relative) != digest:
            raise ValueError(f'Frozen artifact changed: {relative}')


def prepare(project, stage4, stage7, output):
    processed = project/'data/processed'
    baseline, selection = read_json(stage4/'results.json'), read_json(stage7/'results.json')
    for report in [baseline,selection]:
        for name,digest in report['input_sha256'].items():
            if sha256(processed/name) != digest:
                raise ValueError('Historical input hashes do not match')
    train,y,dev,dy = load_development_inputs(processed)
    dev_quality=pd.read_parquet(processed/'quality.parquet').loc[dev.index]
    if selection['protocol']['settings']['seeds'] != SEEDS:
        raise ValueError('Unexpected Stage 7 seeds')
    output.mkdir(parents=True,exist_ok=False)
    roster = []
    hashes = source_hashes(project)
    for name in ['features.parquet','labels.parquet','quality.parquet']:
        path=processed/name; hashes[str(path.relative_to(project))]=sha256(path)

    def register(name, family, seed, path, columns, metadata):
        relative = str(path.relative_to(project))
        hashes[relative] = sha256(path)
        roster.append({'name':name,'family':family,'seed':seed,'path':relative,
            'columns':columns,'metadata':metadata})

    for modality in ['audio','facial','concatenation']:
        for seed in SEEDS:
            name=f'{modality}_seed{seed}'
            path=stage4/'models'/f'{name}.joblib'
            model=joblib.load(path)  # Trusted locally generated artifacts only.
            columns=list(model.named_steps['preprocess'].feature_names_in_)
            old=pd.read_parquet(stage4/f'{name}_dev_predictions.parquet')
            assert old.index.equals(dev.index)
            p=model.predict_proba(dev[columns])[:,list(model.classes_).index(1)]
            np.testing.assert_allclose(p,old.probability_class_1,rtol=1e-6,atol=1e-7)
            register(name,'logistic_'+modality,seed,path,columns,read_json(stage4/f'{name}.json'))
    for selector in ['all','l1','hho']:
        for seed in SEEDS:
            for mode in ['adaptive','quality']:
                name=f'{selector}_{mode}_seed{seed}';path=stage7/f'{name}.pt'
                cp=torch.load(path,map_location='cpu',weights_only=True)
                if cp['train_participant_ids'] != list(map(int,train.index)) or cp['target_policy'] != TARGET_POLICY:
                    raise ValueError('Source checkpoint training IDs/target mismatch')
                restored=predict_checkpoint(path,dev[cp['selected_features']],'cpu',
                    dev_quality[cp['quality_columns']] if 'quality_columns' in cp else None)
                old=pd.read_parquet(stage7/f'{name}_dev_predictions.parquet')
                if not old.index.equals(dev.index):raise ValueError('Historical development ID mismatch')
                for c in restored:
                    np.testing.assert_allclose(restored[c],old[c],rtol=1e-5,atol=1e-6)
                register(name,selector+'_'+mode,seed,path,cp['selected_features'],read_json(stage7/f'{name}.json'))
            source=stage7/f'{selector}_adaptive_seed{seed}.pt'
            cp=torch.load(source,map_location='cpu',weights_only=True)
            parameters={**selection['protocol']['settings']['neural'],'fusion_mode':'equal'}
            logging.info('Training fixed-weight control: %s seed %s',selector,seed)
            new,result,predictions=fit_selected(train,y,dev,dy,cp['selected_features'],parameters,seed,'cpu')
            name=f'{selector}_equal_seed{seed}';path=output/f'{name}.pt'
            torch.save(new,path)
            restored=predict_checkpoint(path,dev[cp['selected_features']],'cpu')
            for c in restored:
                np.testing.assert_allclose(restored[c],predictions[c],rtol=1e-6,atol=1e-7)
            predictions.to_parquet(output/f'{name}_dev_predictions.parquet')
            write_json(output/f'{name}.json',result)
            register(name,selector+'_equal',seed,path,cp['selected_features'],result)
    if len(roster)!=36:raise ValueError('Incomplete model roster')
    manifest={'target_policy':TARGET_POLICY,'threshold':.5,'primary_seed':42,'seeds':SEEDS,
        'test_evaluated':False,'models':roster,'frozen_hashes':hashes,
        'train_ids':list(map(int,train.index)),'dev_ids':list(map(int,dev.index)),
        'majority_class':int(y.value_counts().idxmax()),'bootstrap_repeats':2000,'bootstrap_seed':91042,
        'provisional_label_authorization':'User approved unchanged released labels for final evaluation on 2026-10-28'}
    write_json(output/'frozen_manifest.json',manifest)
    return manifest


def ablation_pairs():
    pairs=[]
    for selector in ['all','l1','hho']:
        pairs += [(selector+'_quality',selector+'_adaptive','add_quality'),
                  (selector+'_adaptive',selector+'_equal','add_gate')]
    for mode in ['equal','adaptive','quality']:
        pairs += [('hho_'+mode,'all_'+mode,'HHO_selection'),('l1_'+mode,'all_'+mode,'L1_selection')]
    return pairs


def figures(output, table, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    subset=summary[summary.partition=='test']
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    ax.barh(subset.family,subset.f1_mean,xerr=subset.f1_sd,color='#3478a8',capsize=3)
    ax.set(xlim=(0,1),xlabel='Positive-class F1 (mean; bars show seed SD, not CI)',
           title='Held-out test: provisional released labels')
    ax.invert_yaxis()
    for suffix in ['png','svg']:fig.savefig(output/f'test_f1.{suffix}',dpi=180)
    plt.close(fig)
    primary=table[(table.partition=='test')&(table.seed==42)]
    fig,axes=plt.subplots(4,4,figsize=(12,12),layout='constrained')
    for ax,(_,row) in zip(axes.flat,primary.iterrows()):
        matrix=np.array([[row.tn,row.fp],[row.fn,row.tp]],dtype=int)
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=56)
        for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,str(v),ha='center',va='center',color='black')
        ax.set(title=row.family,xticks=[0,1],yticks=[0,1],xlabel='Predicted',ylabel='Released label')
    for ax in list(axes.flat)[len(primary):]:ax.set_visible(False)
    fig.suptitle('Provisional test confusion matrices — primary seed 42')
    for suffix in ['png','svg']:fig.savefig(output/f'test_confusion.{suffix}',dpi=180)
    plt.close(fig)


def evaluate(project, run_dir):
    manifest_path=run_dir/'frozen_manifest.json'
    manifest=read_json(manifest_path)
    verify_manifest(manifest,project)
    output=run_dir/'evaluation'
    if output.exists():raise ValueError('Evaluation output already exists; do not overwrite')
    seed_everything(42)
    processed=project/'data/processed'
    features=pd.read_parquet(processed/'features.parquet')
    labels=pd.read_parquet(processed/'labels.parquet')
    quality=pd.read_parquet(processed/'quality.parquet')
    if not features.index.is_unique or not features.index.equals(labels.index) or not quality.index.equals(labels.index):
        raise ValueError('Participant alignment failure')
    if set(labels.columns)!={'split','PHQ_Binary'} or set(labels.split)!={'train','dev','test'}:
        raise ValueError('Unexpected official partition schema')
    if list(map(int,labels.index[labels.split=='train']))!=manifest['train_ids'] or list(map(int,labels.index[labels.split=='dev']))!=manifest['dev_ids']:
        raise ValueError('Frozen partition mismatch')
    output.mkdir()
    rows, predictions, ablations=[],[],[]
    for partition in ['dev','test']:
        ids=labels.index[labels.split==partition]
        x,y=features.loc[ids],labels.loc[ids,'PHQ_Binary']
        validate_partition(x,y)
        indices=bootstrap_indices(y,manifest['bootstrap_repeats'],manifest['bootstrap_seed'])
        scores={}
        for entry in manifest['models'] + [{'name':'majority_seed42','family':'majority','seed':42}]:
            logging.info('Scoring %s: %s',partition,entry['name'])
            started=time.perf_counter();feature_count=0;parameter_count=None
            if entry['family']=='majority':
                p=np.full(len(x),manifest['majority_class'],dtype=float)
            elif entry['family'].startswith('logistic_'):
                model=joblib.load(project/entry['path'])
                p=model.predict_proba(x[entry['columns']])[:,list(model.classes_).index(1)]
                feature_count=len(model.named_steps['preprocess'].columns_)
            else:
                path=project/entry['path'];cp=torch.load(path,map_location='cpu',weights_only=True)
                q=quality.loc[ids,cp['quality_columns']] if 'quality_columns' in cp else None
                p=predict_checkpoint(path,x[entry['columns']],'cpu',q).probability_class_1.to_numpy()
                feature_count=len(cp['preprocessing']['output_columns'])
                parameter_count=sum(v.numel() for k,v in cp['state_dict'].items() if k!='quality_mean')
            seconds=time.perf_counter()-started
            metrics=binary_metrics(y,p)
            tn,fp,fn,tp=np.array(metrics['confusion_matrix']).ravel()
            row={'name':entry['name'],'family':entry['family'],'seed':entry['seed'],'partition':partition,
                 **{m:metrics[m] for m in METRICS},'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
                 'retained_features':feature_count,'neural_parameters':parameter_count,
                 'restore_plus_inference_seconds':seconds}
            if entry['seed']==42:
                samples=bootstrap_f1(y,p,indices)
                row['f1_ci_low'],row['f1_ci_high']=interval(samples)
            else:row['f1_ci_low']=row['f1_ci_high']=None
            rows.append(row);scores[entry['name']]=(metrics['f1'],bootstrap_f1(y,p,indices) if entry['seed']==42 else None)
            predictions.extend({'name':entry['name'],'partition':partition,'participant_id':int(i),
                'PHQ_Binary':int(t),'probability_class_1':float(prob),'prediction':int(prob>=.5)} for i,t,prob in zip(ids,y,p))
        for first,second,effect in ablation_pairs():
            for seed in SEEDS:
                a,b=scores[f'{first}_seed{seed}'],scores[f'{second}_seed{seed}']
                low,high=interval(a[1]-b[1]) if seed==42 else (None,None)
                ablations.append({'partition':partition,'effect':effect,'first':first,'second':second,
                    'seed':seed,'f1_difference':a[0]-b[0],'paired_ci_low':low,'paired_ci_high':high})
    table=pd.DataFrame(rows)
    table.to_csv(output/'all_metrics.csv',index=False)
    pd.DataFrame(predictions).to_parquet(output/'predictions.parquet',index=False)
    pd.DataFrame(ablations).to_csv(output/'paired_ablations.csv',index=False)
    summary=[]
    for (partition,family),group in table.groupby(['partition','family'],sort=True):
        item={'partition':partition,'family':family,'n_seeds':len(group)}
        for metric in METRICS:
            item[metric+'_mean']=float(group[metric].mean());item[metric+'_sd']=float(group[metric].std(ddof=0))
        summary.append(item)
    summary=pd.DataFrame(summary);summary.to_csv(output/'seed_summary.csv',index=False)
    figures(output,table,summary)
    lines=['# Stage 9: provisional final evaluation','',
        'Target: unchanged released PHQ_Binary; unresolved label discrepancies remain.',
        'All models trained on official training participants only. Threshold 0.5. No test-based selection.',
        'Mean and population SD across three seeds; majority has one deterministic prediction.','',
        '| Model | Dev F1 mean | Test F1 mean ± SD | Test precision | Test recall | Test AUC |',
        '|---|---:|---:|---:|---:|---:|']
    for _,row in summary[summary.partition=='test'].iterrows():
        dev=summary[(summary.partition=='dev')&(summary.family==row.family)].iloc[0]
        lines.append(f'| {row.family} | {dev.f1_mean:.3f} | {row.f1_mean:.3f} ± {row.f1_sd:.3f} | {row.precision_mean:.3f} | {row.recall_mean:.3f} | {row.roc_auc_mean:.3f} |')
    lines += ['', 'Primary-seed F1 bootstrap intervals and every confusion matrix are in all_metrics.csv.',
        'Paired ablation intervals are descriptive, conditional on observed prevalence and fitted models.',
        'No correction for multiple comparisons; no clinical or benchmark superiority claim.',
        'Timing includes checkpoint restoration; historical training times in the manifest are not a controlled speed benchmark.',
        'Small sample, whole-interview aggregation, seed sensitivity and unresolved label semantics limit interpretation.']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(output/'results.json',{'test_evaluated':True,'target_policy':TARGET_POLICY,
        'manifest_sha256':sha256(manifest_path),'n_models':37,'n_test':int((labels.split=='test').sum()),
        'threshold':.5,'bootstrap_repeats':2000,'bootstrap_seed':91042,
        'artifact_sha256':{p.name:sha256(p) for p in output.iterdir() if p.is_file()}})
    return table


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--stage4',type=Path,required=True);p.add_argument('--stage7',type=Path,required=True)
    e=sub.add_parser('evaluate');e.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();project=Path(__file__).resolve().parents[2]
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    if args.command=='prepare':
        output=project/'experiments/runs'/('stage9_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        prepare(project,args.stage4.resolve(),args.stage7.resolve(),output)
        logging.info('Frozen evaluation manifest: %s',output)
    else:evaluate(project,args.run.resolve())


if __name__=='__main__':main()
