"""Stage 4: training-CV-selected logistic baselines; development evaluation only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import platform
import subprocess
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
import yaml

from src.dataset.preprocessing import ParticipantPreprocessor
from src.experiments.metrics import binary_metrics

LOG = logging.getLogger(__name__)
TARGET_POLICY = 'released_binary_provisional'
MODALITIES = ('audio', 'facial', 'concatenation')


def validate_partition(X: pd.DataFrame, y: pd.Series) -> None:
    ParticipantPreprocessor._validate(X)
    if not X.index.is_unique or not X.index.equals(y.index) or len(X) == 0:
        raise ValueError('Participant IDs and labels must be unique, nonempty and aligned')
    ids = pd.to_numeric(X.index, errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(ids).all() or (ids <= 0).any() or (ids % 1 != 0).any():
        raise ValueError('Invalid participant IDs')
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError('Target must contain only nonmissing binary labels')


def fit_baseline(X_train: pd.DataFrame, y_train: pd.Series,
                 X_dev: pd.DataFrame, y_dev: pd.Series,
                 modality: str, seed: int, folds: int, c_values: list[float],
                 class_weights: list) -> tuple[Pipeline, dict, pd.DataFrame]:
    """Select parameters using train CV only; fixed 0.5 threshold on dev.

    There is intentionally no test-data argument. Validation labels are used
    exclusively after selection/refitting, when calculating reported metrics.
    """
    validate_partition(X_train, y_train)
    validate_partition(X_dev, y_dev)
    if len(X_train.index.intersection(X_dev.index)):
        raise ValueError('Participant overlap between train and development')
    if list(X_train.columns) != list(X_dev.columns):
        raise ValueError('Training/development schemas differ')
    if modality not in MODALITIES:
        raise ValueError('Unknown modality')
    if folds < 2 or y_train.nunique() != 2 or y_train.value_counts().min() < folds:
        raise ValueError('Insufficient training examples per class for stratified folds')
    if not c_values or any(not np.isfinite(c) or c <= 0 for c in c_values):
        raise ValueError('Regularization C values must be finite and positive')
    if not class_weights or any(w not in (None, 'balanced') for w in class_weights):
        raise ValueError('Supported class weights: null or balanced')
    columns = [c for c in X_train if modality == 'concatenation' or c.startswith(modality + '__')]
    if not columns:
        raise ValueError('No descriptors for selected modality')
    train, dev = X_train[columns], X_dev[columns]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(train, y_train))
    pipeline = Pipeline([
        ('preprocess', ParticipantPreprocessor()),
        ('classifier', LogisticRegression(solver='liblinear', max_iter=5000, random_state=seed)),
    ])
    search = GridSearchCV(
        pipeline, {'classifier__C': c_values, 'classifier__class_weight': class_weights},
        scoring='f1', cv=splits, refit=True, n_jobs=1, error_score='raise',
        return_train_score=False,
    )
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        search.fit(train, y_train)
    selection_seconds = time.perf_counter() - started
    model = search.best_estimator_
    started = time.perf_counter()
    probability = model.predict_proba(dev)[:, list(model.classes_).index(1)]
    inference_seconds = time.perf_counter() - started
    cv = search.cv_results_
    candidates = []
    for index, params in enumerate(cv['params']):
        candidates.append({
            'parameters': params, 'mean_selection_f1': float(cv['mean_test_score'][index]),
            'std_selection_f1': float(cv['std_test_score'][index]),
            'fold_f1': [float(cv[f'split{fold}_test_score'][index]) for fold in range(folds)],
        })
    preprocess = model.named_steps['preprocess']
    result = {
        'experiment': modality, 'seed': seed, 'evaluation_partition': 'dev',
        'target_policy': TARGET_POLICY, 'original_feature_count': len(columns),
        'retained_feature_count': len(preprocess.columns_),
        'retained_feature_names': preprocess.columns_, 'dropped_train_features': preprocess.dropped_,
        'selected_parameters': search.best_params_,
        'selection_cv_mean_f1': float(search.best_score_),
        'selection_cv_std_f1': float(cv['std_test_score'][search.best_index_]),
        'selection_scores_are_not_unbiased_performance_estimates': True,
        'cv_candidates': candidates,
        'cv_participants': [{'train': list(map(int, train.index[a])), 'validation': list(map(int, train.index[b]))} for a,b in splits],
        'selection_and_refit_seconds': selection_seconds,
        'refit_seconds': float(search.refit_time_), 'inference_seconds': inference_seconds,
        'dev_metrics': binary_metrics(y_dev, probability),
    }
    predictions = pd.DataFrame({'PHQ_Binary': y_dev, 'probability_class_1': probability,
                                'prediction': (probability >= 0.5).astype(int)}, index=y_dev.index)
    return model, result, predictions


def majority_baseline(y_train: pd.Series, y_dev: pd.Series) -> tuple[dict, pd.DataFrame]:
    model = DummyClassifier(strategy='most_frequent')
    started = time.perf_counter()
    model.fit(np.zeros((len(y_train), 1)), y_train)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    probability = model.predict_proba(np.zeros((len(y_dev), 1)))[:, list(model.classes_).index(1)]
    inference_seconds = time.perf_counter() - started
    return {'experiment': 'majority', 'evaluation_partition': 'dev',
            'majority_class': int(model.predict([[0]])[0]), 'refit_seconds': fit_seconds,
            'inference_seconds': inference_seconds, 'original_feature_count': 0,
            'retained_feature_count': 0, 'dev_metrics': binary_metrics(y_dev, probability)}, pd.DataFrame(
                {'PHQ_Binary': y_dev, 'probability_class_1': probability,
                 'prediction': (probability >= 0.5).astype(int)}, index=y_dev.index)


def load_development_inputs(processed: Path) -> tuple:
    """Read raw summaries, never the globally normalized official-split tables."""
    features = pd.read_parquet(processed / 'features.parquet')
    labels = pd.read_parquet(processed / 'labels.parquet')
    if not features.index.is_unique or not labels.index.is_unique or not features.index.equals(labels.index):
        raise ValueError('Feature/label participant alignment failure')
    if set(labels.columns) != {'PHQ_Binary', 'split'} or set(labels['split']) != {'train', 'dev', 'test'}:
        raise ValueError('Unexpected labels/split schema')
    train_ids = labels.index[labels['split'] == 'train']
    dev_ids = labels.index[labels['split'] == 'dev']
    # Test feature values and targets are not passed into the experiment API.
    return features.loc[train_ids], labels.loc[train_ids, 'PHQ_Binary'], features.loc[dev_ids], labels.loc[dev_ids, 'PHQ_Binary']


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(processed: Path, output: Path, settings: dict, project: Path) -> dict:
    if settings.get('target_policy') != TARGET_POLICY:
        raise ValueError('Explicit released_binary_provisional target policy is required')
    seeds = settings['seeds']
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Specify distinct seeds; the first is the predeclared primary seed')
    train, y_train, dev, y_dev = load_development_inputs(processed)
    validate_partition(train, y_train)
    validate_partition(dev, y_dev)
    if y_train.nunique() != 2:
        raise ValueError('Training must contain both classes')
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite previous experiments.
    models = output / 'models'; models.mkdir()
    (output/'protocol.json').write_text(json.dumps({
        'target': 'PHQ_Binary unchanged as supplied', 'target_policy': TARGET_POLICY,
        'evaluation': 'development only; test unevaluated', 'primary_seed': seeds[0],
        'threshold': 0.5, 'selection': 'positive-class F1; stratified participant CV on train only',
        'preprocessing': 'fit separately inside each fold; final refit on official train only',
        'settings': settings,
    }, indent=2), encoding='utf-8')
    results = []
    majority, predictions = majority_baseline(y_train, y_dev)
    predictions.to_parquet(output/'majority_dev_predictions.parquet')
    (output/'majority.json').write_text(json.dumps(majority,indent=2),encoding='utf-8')
    for modality in MODALITIES:
        for seed in seeds:
            LOG.info('Fitting %s, CV seed %s', modality, seed)
            model, result, predictions = fit_baseline(train, y_train, dev, y_dev, modality, seed,
                settings['folds'], settings['c_values'], settings['class_weights'])
            name = f'{modality}_seed{seed}'
            joblib.dump(model, models/f'{name}.joblib')
            predictions.to_parquet(output/f'{name}_dev_predictions.parquet')
            (output/f'{name}.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
            results.append(result)
    summary = []
    for modality in MODALITIES:
        runs = [r for r in results if r['experiment'] == modality]
        row = {'experiment': modality, 'seeds': seeds,
               'original_feature_count': runs[0]['original_feature_count']}
        for metric in ['precision','recall','f1','macro_f1','roc_auc','accuracy']:
            values = [r['dev_metrics'][metric] for r in runs]
            row[f'{metric}_mean'] = float(np.mean(values))
            row[f'{metric}_sd'] = float(np.std(values, ddof=0))
        summary.append(row)
    versions = {p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','joblib','pyarrow']}
    def git(*args):
        completed = subprocess.run(['git', *args], cwd=project, capture_output=True, text=True)
        return completed.stdout.strip() if completed.returncode == 0 else None
    source_paths = [project/'src/experiments/baselines.py',project/'src/experiments/metrics.py',project/'src/dataset/preprocessing.py']
    report = {
        'status': 'provisional_development_baselines', 'test_evaluated': False,
        'primary_seed': seeds[0], 'majority': majority, 'results': results, 'across_seed_summary': summary,
        'environment': {'python': platform.python_version(), 'platform': platform.platform(), 'packages': versions},
        'git_commit': git('rev-parse','HEAD'), 'git_status': git('status','--short'),
        'input_sha256': {p.name: sha256(p) for p in [processed/'features.parquet',processed/'labels.parquet']},
        'source_sha256': {str(p.relative_to(project)): sha256(p) for p in source_paths},
        'limitations': [
            'Released PHQ_Binary is not reconciled with the PHQ-8 >=10 definition; scores are provisional.',
            'Three CV seeds measure sensitivity to the selection folds, not independent cohorts or confidence intervals.',
            'Development metrics are for iteration; the held-out test must remain reserved for a frozen protocol.',
            'Whole-session aggregation does not isolate interviewer speech or preserve temporal dynamics.',
            'No direct comparison to reported DAIC-WOZ benchmark scores: dataset, target and protocol differ.',
            'Checkpoints use joblib; load only trusted locally generated files.',
        ],
    }
    (output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    table = []
    for result in [majority] + [r for r in results if r['seed'] == seeds[0]]:
        table.append({'experiment':result['experiment'], 'features':result['retained_feature_count'], **{k:result['dev_metrics'][k] for k in ['precision','recall','f1','roc_auc','accuracy']},
                      'refit_seconds':result['refit_seconds'],'inference_seconds':result['inference_seconds']})
    pd.DataFrame(table).to_csv(output/'comparison.csv',index=False)
    pd.DataFrame(summary).to_csv(output/'seed_variability.csv',index=False)
    lines = ['# Stage 4: provisional development baselines', '',
        f'Primary CV seed: {seeds[0]}. Threshold: 0.5. Target: supplied PHQ_Binary.',
        'No test evaluation. Positive class = 1. All model selection and preprocessing fitting use training participants only.', '',
        '| Model | Features | Precision | Recall | F1 | ROC-AUC | Accuracy |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for row in table:
        lines.append(f"| {row['experiment']} | {row['features']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['roc_auc']:.3f} | {row['accuracy']:.3f} |")
    lines += ['', '## Limitations', ''] + ['- '+item for item in report['limitations']]
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('configs/default.yaml'))
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    processed = project / config['paths']['processed']
    output = project / 'experiments/runs' / f'stage4_{timestamp}'
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    report = run(processed,output,config['baselines'],project)
    LOG.info('Completed %d fits; results: %s',len(report['results']),output)


if __name__ == '__main__':
    main()
