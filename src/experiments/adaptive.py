"""Stage 5: train-CV selection of compact adaptive fusion; dev evaluation only."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import platform
import random
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
import yaml
from src.models.adaptive import AdaptiveFusion
from src.dataset.preprocessing import ParticipantPreprocessor
from src.experiments.baselines import load_development_inputs, validate_partition, TARGET_POLICY, sha256
from src.experiments.metrics import binary_metrics

LOG = logging.getLogger(__name__)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)


def resolve_device(requested: str) -> str:
    if requested not in ('auto', 'cpu', 'cuda'):
        raise ValueError('Device must be auto, cpu, or cuda')
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    return requested


def synchronize(device: str):
    if device == 'cuda':
        torch.cuda.synchronize()


def tensors(frame: pd.DataFrame, device: str):
    audio = [c for c in frame if c.startswith('audio__')]
    facial = [c for c in frame if c.startswith('facial__')]
    if not audio or not facial:
        raise ValueError('Both modalities must retain at least one training feature')
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError('Nonfinite preprocessed descriptors')
    return (torch.tensor(frame[audio].to_numpy(), dtype=torch.float32, device=device),
            torch.tensor(frame[facial].to_numpy(), dtype=torch.float32, device=device))


def train_network(frame: pd.DataFrame, target: pd.Series, parameters: dict, seed: int, device: str):
    if not frame.index.equals(target.index) or set(target.unique()) != {0, 1}:
        raise ValueError('Aligned binary training data with both classes required')
    if parameters['epochs'] < 1 or parameters['learning_rate'] <= 0 or parameters['weight_decay'] < 0:
        raise ValueError('Invalid optimizer settings')
    seed_everything(seed)
    audio, facial = tensors(frame, device)
    config = {'audio_dim':audio.shape[1], 'facial_dim':facial.shape[1],
              'hidden_dim':parameters['hidden_dim'], 'dropout':parameters['dropout']}
    model = AdaptiveFusion(**config).to(device)
    labels = torch.tensor(target.to_numpy(), dtype=torch.float32, device=device)
    pos_weight = float((target == 0).sum() / (target == 1).sum())
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight,device=device))
    optimizer = torch.optim.AdamW(model.parameters(),lr=parameters['learning_rate'],weight_decay=parameters['weight_decay'])
    history = []
    model.train()
    for epoch in range(parameters['epochs']):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(audio,facial)
        loss = loss_fn(logits,labels)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite training loss')
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach().cpu()))
    model.eval()
    return model, config, history, pos_weight


def predict(model: AdaptiveFusion, frame: pd.DataFrame, device: str):
    model.eval()
    with torch.no_grad():
        logits, weights = model(*tensors(frame,device))
        return torch.sigmoid(logits).cpu().numpy(), weights.cpu().numpy()


def preprocessing_state(p: ParticipantPreprocessor) -> dict:
    return {'input_columns':list(p.feature_names_in_), 'output_columns':p.columns_,
            'dropped':p.dropped_, 'medians':p.medians_.to_dict(),
            'means':p.means_.to_dict(), 'scales':p.scales_.to_dict()}


def fit_adaptive(train: pd.DataFrame, y: pd.Series, dev: pd.DataFrame, dy: pd.Series,
                 seed: int, folds: int, candidates: list[dict], device: str):
    validate_partition(train,y)
    validate_partition(dev,dy)
    if len(train.index.intersection(dev.index)) or list(train.columns) != list(dev.columns):
        raise ValueError('Overlapping participants or incompatible schemas')
    if folds < 2 or y.nunique() != 2 or y.value_counts().min() < folds or not candidates:
        raise ValueError('Insufficient training data or no candidate settings')
    device = resolve_device(device)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    synchronize(device)
    started = time.perf_counter()
    splits = list(StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed).split(train,y))
    fold_data = []
    for a,b in splits:
        prep = ParticipantPreprocessor().fit(train.iloc[a])
        fold_data.append((prep.transform(train.iloc[a]), y.iloc[a], prep.transform(train.iloc[b]), y.iloc[b]))
    selection = []
    for candidate in candidates:
        scores = []
        for fold, (X_fit,y_fit,X_val,y_val) in enumerate(fold_data):
            model,_,_,_ = train_network(X_fit,y_fit,candidate,seed+fold+1,device)
            probability,_ = predict(model,X_val,device)
            scores.append(binary_metrics(y_val,probability)['f1'])
            del model
        selection.append({'parameters':candidate, 'fold_f1':scores,
                          'mean_f1':float(np.mean(scores)), 'sd_f1':float(np.std(scores))})
    best_index = int(np.argmax([r['mean_f1'] for r in selection]))
    best = selection[best_index]
    prep = ParticipantPreprocessor().fit(train)
    X_fit = prep.transform(train)
    synchronize(device)
    refit_started = time.perf_counter()
    model, config, history, pos_weight = train_network(X_fit,y,best['parameters'],seed,device)
    synchronize(device)
    refit_seconds = time.perf_counter()-refit_started
    fit_seconds = time.perf_counter()-started
    inference_started = time.perf_counter()
    probability, weights = predict(model,prep.transform(dev),device)
    synchronize(device)
    inference_seconds = time.perf_counter()-inference_started
    result = {
        'experiment':'adaptive_fusion', 'seed':seed, 'target_policy':TARGET_POLICY,
        'evaluation_partition':'dev', 'test_evaluated':False, 'device':device,
        'selected_parameters':best['parameters'], 'cv_candidates':selection,
        'selection_cv_mean_f1':best['mean_f1'], 'selection_cv_sd_f1':best['sd_f1'],
        'selection_scores_are_not_unbiased_performance_estimates':True,
        'cv_participants':[{'train':list(map(int,train.index[a])), 'validation':list(map(int,train.index[b]))} for a,b in splits],
        'original_feature_count':len(train.columns), 'retained_feature_count':len(prep.columns_),
        'retained_feature_names':prep.columns_, 'parameter_count':sum(p.numel() for p in model.parameters()),
        'model_config':config, 'train_positive_weight':pos_weight,
        'training_loss_by_epoch':history, 'selection_and_refit_seconds':fit_seconds,
        'refit_seconds':refit_seconds, 'inference_seconds':inference_seconds,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if device=='cuda' else None,
        'dev_metrics':binary_metrics(dy,probability),
        'gate_summary':{m:{'mean':float(weights[:,i].mean()),'min':float(weights[:,i].min()),'max':float(weights[:,i].max()),'sd':float(weights[:,i].std())} for i,m in enumerate(['audio','facial'])},
    }
    predictions = pd.DataFrame({'PHQ_Binary':dy, 'probability_class_1':probability,
        'prediction':(probability>=0.5).astype(int), 'audio_weight':weights[:,0], 'facial_weight':weights[:,1]},index=dev.index)
    checkpoint = {'model_config':config,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                  'preprocessing':preprocessing_state(prep), 'train_participant_ids':list(map(int,train.index)),
                  'seed':seed, 'target_policy':TARGET_POLICY, 'threshold':0.5}
    return checkpoint,result,predictions


def predict_checkpoint(path: Path, raw: pd.DataFrame, device: str='cpu') -> pd.DataFrame:
    device = resolve_device(device)
    saved = torch.load(path,map_location='cpu',weights_only=True)
    prep = ParticipantPreprocessor.from_state(saved['preprocessing'])
    model = AdaptiveFusion(**saved['model_config']).to(device)
    model.load_state_dict(saved['state_dict'])
    probability,weights = predict(model,prep.transform(raw),device)
    return pd.DataFrame({'probability_class_1':probability,'audio_weight':weights[:,0],'facial_weight':weights[:,1]},index=raw.index)


def run(processed: Path, output: Path, settings: dict, project: Path):
    if settings.get('target_policy') != TARGET_POLICY:
        raise ValueError('Explicit provisional target policy required')
    seeds = settings['seeds']
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Distinct predeclared seeds required')
    device = resolve_device(settings['device'])
    train,y,dev,dy = load_development_inputs(processed)
    candidates = [{'hidden_dim':settings['hidden_dim'], 'dropout':settings['dropout'],
        'learning_rate':settings['learning_rate'],'epochs':epochs,'weight_decay':decay}
        for epochs in settings['epochs'] for decay in settings['weight_decays']]
    output.mkdir(parents=True,exist_ok=False)
    protocol = {'settings':settings,'device':device,'threshold':0.5,'primary_seed':seeds[0],
        'selection':'positive-class F1, training-only participant CV; first candidate wins ties',
        'target_policy':TARGET_POLICY,'test_evaluated':False,'quality_inputs':False,
        'candidates':candidates,'class_imbalance':'BCE positive weight = negatives/positives in each fitting fold',
        'stopping':'fixed candidate epoch counts selected by train CV; no development early stopping'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    results=[]
    for seed in seeds:
        LOG.info('Adaptive fusion on %s, seed %s',device,seed)
        checkpoint,result,predictions = fit_adaptive(train,y,dev,dy,seed,settings['folds'],candidates,device)
        name=f'adaptive_seed{seed}'
        torch.save(checkpoint,output/f'{name}.pt')
        predictions.to_parquet(output/f'{name}_dev_predictions.parquet')
        (output/f'{name}.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        restored=predict_checkpoint(output/f'{name}.pt',dev,device)
        for column in restored:
            np.testing.assert_allclose(restored[column],predictions[column],rtol=1e-6,atol=1e-7)
        result['checkpoint_verified']=True
        results.append(result)
    summary={}
    for metric in ['precision','recall','f1','roc_auc','accuracy']:
        values=[r['dev_metrics'][metric] for r in results]
        summary[metric]={'mean':float(np.mean(values)),'sd':float(np.std(values))}
    report={'protocol':protocol,'results':results,'across_seed_summary':summary,
        'environment':{'python':platform.python_version(),'torch':str(torch.__version__),'cuda_runtime':torch.version.cuda,
            'device':device,'gpu':torch.cuda.get_device_name() if device=='cuda' else None},
        'input_sha256':{n:sha256(processed/n) for n in ['features.parquet','labels.parquet']},
        'source_sha256':{str(p.relative_to(project)):sha256(p) for p in [Path(__file__),project/'src/models/adaptive.py',project/'src/dataset/preprocessing.py']},
        'limitations':['Provisional released labels; not a verified PHQ-8 threshold experiment.',
            'Gate weights are learned mixing coefficients, not causal explanations or calibrated reliability.',
            'No explicit quality inputs; reliability-aware fusion belongs to Stage 6.',
            'Development is used for iteration; test is unevaluated.',
            'Seed variability is not a confidence interval; same development participants across runs.',
            'Session summaries do not isolate interviewer speech or preserve temporal dynamics.',
            'Exact reproducibility is checked on this software/device; cross-device results may differ.']}
    (output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    table=[{'seed':r['seed'],**{k:r['dev_metrics'][k] for k in ['precision','recall','f1','roc_auc','accuracy']},'features':r['retained_feature_count'],'parameters':r['parameter_count'],'refit_seconds':r['refit_seconds'],'inference_seconds':r['inference_seconds']} for r in results]
    pd.DataFrame(table).to_csv(output/'comparison.csv',index=False)
    lines=['# Stage 5: provisional adaptive fusion development results','',f'Primary seed: {seeds[0]}. Threshold: 0.5. Device: {device}. Test unevaluated.','',
        '| Seed | Precision | Recall | F1 | ROC-AUC | Accuracy |','|---|---:|---:|---:|---:|---:|']
    for row in table:
        lines.append('| '+str(row['seed'])+' | '+' | '.join(f"{row[k]:.3f}" for k in ['precision','recall','f1','roc_auc','accuracy'])+' |')
    lines += ['', '## Limitations','']+['- '+s for s in report['limitations']]
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/default.yaml'))
    args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text(encoding='utf-8'))
    project=Path(__file__).resolve().parents[2]
    output=project/'experiments/runs'/('stage5_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    run(project/config['paths']['processed'],output,config['adaptive'],project)
    LOG.info('Completed: %s',output)


if __name__=='__main__':
    main()
