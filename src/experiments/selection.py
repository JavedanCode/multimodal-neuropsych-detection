"""Stage 7: bounded Binary HHO and L1, evaluated with fixed neural settings."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import logging
from pathlib import Path
import platform
import time
import numpy as np
import pandas as pd
import torch
import yaml
from src.selection.hho import binary_hho
from src.selection.fitness import TrainingFitness,select_l1
from src.dataset.preprocessing import ParticipantPreprocessor
from src.dataset.quality import QUALITY_COLUMNS,validate_quality
from src.experiments.baselines import load_development_inputs,TARGET_POLICY,sha256
from src.experiments.adaptive import train_network,predict,predict_checkpoint,preprocessing_state,resolve_device,synchronize
from src.experiments.metrics import binary_metrics

LOG=logging.getLogger(__name__)


def fit_selected(train,y,dev,dy,columns,parameters,seed,device,train_quality=None,dev_quality=None):
    from src.experiments.baselines import validate_partition
    validate_partition(train,y);validate_partition(dev,dy)
    if len(train.index.intersection(dev.index)) or list(train)!=list(dev):
        raise ValueError('Overlapping participants or schema mismatch')
    if not columns or len(set(columns))!=len(columns) or not set(columns).issubset(train.columns):
        raise ValueError('Invalid selected descriptor names')
    if (train_quality is None)!=(dev_quality is None):raise ValueError('Both quality tables required')
    if train_quality is not None:
        validate_quality(train_quality,train.index);validate_quality(dev_quality,dev.index)
    synchronize(device);started=time.perf_counter()
    prep=ParticipantPreprocessor().fit(train[columns])
    model,config,history,pos_weight=train_network(prep.transform(train[columns]),y,parameters,seed,device,train_quality)
    synchronize(device);training_seconds=time.perf_counter()-started
    started=time.perf_counter()
    probability,weights=predict(model,prep.transform(dev[columns]),device,dev_quality)
    synchronize(device);inference_seconds=time.perf_counter()-started
    checkpoint={'model_config':config,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
        'preprocessing':preprocessing_state(prep),'train_participant_ids':list(map(int,train.index)),
        'selected_features':columns,'seed':seed,'target_policy':TARGET_POLICY,'threshold':.5}
    if train_quality is not None:checkpoint['quality_columns']=QUALITY_COLUMNS
    result={'seed':seed,'quality_inputs':train_quality is not None,'original_feature_count':len(train.columns),
        'selected_feature_count':len(columns),'retained_feature_count':len(prep.columns_),
        'selected_features':[{'name':c,'modality':c.split('__',1)[0]} for c in columns],
        'retained_feature_names':prep.columns_,'parameter_count':sum(p.numel() for p in model.parameters()),
        'parameters':parameters,'train_positive_weight':pos_weight,'training_seconds':training_seconds,
        'inference_seconds':inference_seconds,'training_loss_by_epoch':history,'dev_metrics':binary_metrics(dy,probability),
        'test_evaluated':False,'target_policy':TARGET_POLICY}
    predictions=pd.DataFrame({'PHQ_Binary':dy,'probability_class_1':probability,'prediction':(probability>=.5).astype(int),
        'audio_weight':weights[:,0],'facial_weight':weights[:,1]},index=dev.index)
    return checkpoint,result,predictions


def run(processed: Path,output: Path,settings: dict,project: Path):
    if settings['target_policy']!=TARGET_POLICY:raise ValueError('Provisional target policy required')
    seeds=settings['seeds']
    if not seeds or len(set(seeds))!=len(seeds):raise ValueError('Distinct predeclared seeds required')
    device=resolve_device(settings['device'])
    train,y,dev,dy=load_development_inputs(processed)
    q=pd.read_parquet(processed/'quality.parquet');labels=pd.read_parquet(processed/'labels.parquet')
    if not q.index.is_unique or not q.index.equals(labels.index) or set(q)!=set(QUALITY_COLUMNS):raise ValueError('Quality alignment/schema mismatch')
    qt,qd=q.loc[train.index,QUALITY_COLUMNS],q.loc[dev.index,QUALITY_COLUMNS]
    validate_quality(qt,train.index);validate_quality(qd,dev.index)
    output.mkdir(parents=True,exist_ok=False)
    protocol={'settings':settings,'device':device,'primary_seed':seeds[0],'test_evaluated':False,
        'selection':'HHO uses train-only CV logistic F1; L1 fits train only with fixed C',
        'downstream':'fixed neural hyperparameters; no post-selection CV or development tuning',
        'threshold':.5,'hho_variant':'sigmoid binary transfer, bounded continuous HHO proposals, paired mask/score state, modality repair'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    results=[];selections=[]
    groups=[np.array([i for i,c in enumerate(train) if c.startswith(m+'__')]) for m in ['audio','facial']]
    for seed in seeds:
        LOG.info('Selecting descriptors: seed %s',seed)
        fitness=TrainingFitness(train,y,seed,settings['folds'],settings['alpha'],settings['beta'],settings['fitness_C'])
        start=time.perf_counter()
        hho=binary_hho(fitness,len(train.columns),groups,seed,settings['population'],settings['iterations'])
        hho['seconds']=time.perf_counter()-start
        hho['unique_masks_evaluated']=len(fitness.records)
        hho['candidate_records']=fitness.records
        hho['cv_participants']=[{'train':list(map(int,train.index[a])),'validation':list(map(int,train.index[b]))} for a,b in fitness.splits]
        hho['seed']=seed
        if hho['fitness']>=2:raise ValueError('HHO did not find a valid subset')
        l1=select_l1(train,y,seed,settings['l1_C'])
        selection={'seed':seed,'hho':hho,'l1':l1}
        (output/f'selection_seed{seed}.json').write_text(json.dumps(selection,indent=2,allow_nan=False),encoding='utf-8')
        pd.DataFrame(hho['history']).to_csv(output/f'hho_seed{seed}_convergence.csv',index=False)
        selections.append(selection)
        masks={'all':np.ones(len(train.columns),dtype=bool),'l1':np.array(l1['mask']),'hho':np.array(hho['mask'])}
        for method,mask in masks.items():
            columns=train.columns[mask].tolist()
            selected_time=0. if method=='all' else selection[method]['seconds']
            for quality_aware in [False,True]:
                model_name='quality' if quality_aware else 'adaptive'
                name=f'{method}_{model_name}_seed{seed}'
                LOG.info('Fitting %s (%d descriptors)',name,len(columns))
                cp,result,predictions=fit_selected(train,y,dev,dy,columns,settings['neural'],seed,device,qt if quality_aware else None,qd if quality_aware else None)
                result.update(selector=method,model=model_name,selection_seconds=selected_time,
                    selection_plus_training_seconds=selected_time+result['training_seconds'])
                torch.save(cp,output/f'{name}.pt')
                predictions.to_parquet(output/f'{name}_dev_predictions.parquet')
                restored=predict_checkpoint(output/f'{name}.pt',dev[columns],device,qd if quality_aware else None)
                for c in restored:np.testing.assert_allclose(restored[c],predictions[c],rtol=1e-6,atol=1e-7)
                result['checkpoint_verified']=True
                (output/f'{name}.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
                results.append(result)
    summary=[]
    for selector in ['all','l1','hho']:
        for model in ['adaptive','quality']:
            rows=[r for r in results if r['selector']==selector and r['model']==model]
            item={'selector':selector,'model':model}
            for metric in ['f1','precision','recall','roc_auc','accuracy']:
                values=[r['dev_metrics'][metric] for r in rows]
                item[metric+'_mean']=float(np.mean(values));item[metric+'_sd']=float(np.std(values))
            item['selected_counts']=[r['selected_feature_count'] for r in rows];summary.append(item)
    report={'protocol':protocol,'results':results,'across_seed_summary':summary,
        'input_sha256':{n:sha256(processed/n) for n in ['features.parquet','labels.parquet','quality.parquet']},
        'source_sha256':{str(p.relative_to(project)):sha256(p) for p in [Path(__file__),project/'src/selection/hho.py',project/'src/selection/fitness.py',project/'src/experiments/adaptive.py',project/'src/dataset/preprocessing.py']},
        'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'device':device,'gpu':torch.cuda.get_device_name() if device=='cuda' else None},
        'limitations':['Small predeclared HHO budget; no claim of optimal masks or novelty.',
            'CV fitness is a selection statistic, not unbiased held-out performance.',
            'Logistic fitness is a surrogate, not neural fusion F1; selected masks may not transfer well.',
            'Neural settings are fixed here, unlike Stage 5/6 CV selection: use these matched all-feature controls.',
            'Same development participants across seeds; SD is not a confidence interval.',
            'Selection time is incurred once per method/seed and reused by both models; do not sum duplicated selection time.',
            'Released PHQ_Binary semantics unresolved; no clinical or benchmark superiority claim.']}
    (output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    table=[{'selector':r['selector'],'model':r['model'],'seed':r['seed'],'selected':r['selected_feature_count'],
        **{m:r['dev_metrics'][m] for m in ['precision','recall','f1','roc_auc','accuracy']},
        'selection_seconds':r['selection_seconds'],'training_seconds':r['training_seconds'],'inference_seconds':r['inference_seconds']} for r in results]
    pd.DataFrame(table).to_csv(output/'comparison.csv',index=False)
    pd.DataFrame(summary).to_csv(output/'seed_summary.csv',index=False)
    lines=['# Stage 7: provisional feature-selection experiments','',f'Primary seed {seeds[0]}; fixed downstream settings; test unevaluated.','',
        '| Selector | Model | Features | Precision | Recall | F1 | ROC-AUC |','|---|---|---:|---:|---:|---:|---:|']
    for row in table:
        if row['seed']==seeds[0]:lines.append(f"| {row['selector']} | {row['model']} | {row['selected']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['roc_auc']:.3f} |")
    lines+=['','## Limitations','']+['- '+s for s in report['limitations']]
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=Path('configs/default.yaml'))
    args=parser.parse_args();config=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    project=Path(__file__).resolve().parents[2]
    output=project/'experiments/runs'/('stage7_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    run(project/config['paths']['processed'],output,config['feature_selection'],project)
    LOG.info('Completed: %s',output)


if __name__=='__main__':main()
