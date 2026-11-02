"""Nested text representation/threshold search and text-dominant late fusion."""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from threadpoolctl import threadpool_limits

from src.dataset.preprocessing import ParticipantPreprocessor
from src.models.text_preprocessing import TextPreprocessor
from src.experiments.audio_study import SEEDS, estimator
from src.experiments.baselines import sha256, TARGET_POLICY
from src.experiments.text_temporal_study import load, summarize


def features(project):
    sets, values, y = load(project)
    mean = sets['text']
    bins, mask = values['text'], values['masks'][:, 0]
    if not mask.any(1).all():
        raise ValueError('Missing screened text')
    sd = np.stack([x[m].std(0) for x, m in zip(bins, mask)])
    maximum = np.stack([x[m].max(0) for x, m in zip(bins, mask)])
    rich = pd.DataFrame(np.concatenate([mean.to_numpy(), sd, maximum], 1), index=y.index,
                        columns=[f'text__rich{i}' for i in range(1152)])
    return {'mean': mean, 'rich': rich, 'audio': sets['audio_only'], 'face': sets['face_only']}, y


def preprocessing(kind):
    return {'standard': StandardScaler, 'l2': Normalizer, 'pca': TextPreprocessor,
            'base': ParticipantPreprocessor}[kind]()


def grid(x, y, splits, scaling, families, thresholds, seed):
    prepared = []
    for a, b in splits:
        prep = preprocessing(scaling).fit(x.iloc[a])
        prepared.append((a, b, prep.transform(x.iloc[a]), prep.transform(x.iloc[b])))
    records = []
    for family in families:
        for c in [.01, .1, 1.]:
            for weight in [None, 'balanced']:
                params = {'C': c, 'class_weight': weight}
                if family == 'rbf':
                    params['gamma'] = 'scale'
                scores = np.full(len(y), np.nan)
                for a, b, fit, valid in prepared:
                    model = estimator(family, params, seed).fit(fit, y.iloc[a])
                    scores[b] = model.decision_function(valid)
                if not np.isfinite(scores).all():
                    raise ValueError('Invalid inner scores')
                for threshold in thresholds:
                    records.append({'family': family, 'parameters': params, 'scaling': scaling,
                                    'threshold': threshold, 'inner_f1': float(f1_score(y, scores >= threshold, zero_division=0)),
                                    'inner_scores': scores.tolist()})
    return records


def select(records, default=False):
    return max([r for r in records if not default or r['threshold'] == 0], key=lambda r:r['inner_f1'])


def fit_candidate(x, y, specification, seed):
    return Pipeline([('preprocess', preprocessing(specification['scaling'])),
                     ('classifier', estimator(specification['family'], specification['parameters'], seed))]).fit(x, y)


def probabilities(scores):
    # LogisticRegression decision margins map to class-one probabilities.
    return 1 / (1 + np.exp(-np.clip(scores, -700, 700)))


def fusion_search(branches, y):
    probs = np.stack([probabilities(b['inner_scores']) for b in branches], 1)
    records = []
    for text_weight in [1., .9, .75, .5]:
        weights = [text_weight, (1-text_weight)/2, (1-text_weight)/2]
        raw = probs @ weights
        for threshold in [.5, .4, .6]:
            records.append({'weights': weights, 'threshold': threshold,
                            'inner_f1': float(f1_score(y, raw >= threshold, zero_division=0)),
                            'inner_scores': raw.tolist()})
    return max(records, key=lambda r:r['inner_f1']), records


def predict(bundle, data):
    if bundle['kind'] == 'text':
        raw = bundle['model'].decision_function(data[bundle['feature']])
    else:
        raw = np.stack([model.predict_proba(data[name])[:, 1]
                        for name, model in zip(['mean', 'audio', 'face'], bundle['models'])], 1) @ bundle['weights']
    return raw, raw-bundle['threshold']


def run(project):
    data, y = features(project)
    out = project/'experiments/runs'/('text_refinement_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir(); (out/'models').mkdir()
    sources = ['docs/text_refinement_protocol.md', 'src/experiments/text_refinement.py',
               'src/experiments/text_temporal_study.py', 'src/experiments/audio_study.py',
               'src/dataset/preprocessing.py', 'src/models/text_preprocessing.py',
               'src/dataset/text_temporal.py', 'src/experiments/metrics.py', 'src/experiments/baselines.py',
               'data/processed/text_temporal_v1/sequences.npz', 'data/processed/text_temporal_v1/manifest.json',
               'data/processed/text_temporal_v1/labels.parquet', 'data/processed/features.parquet', 'data/processed/labels.parquet']
    hashes = {p:sha256(project/p) for p in sources}
    (out/'protocol.json').write_text(json.dumps({'hashes': hashes, 'target': TARGET_POLICY,
        'participants': list(map(int, y.index)), 'seeds': SEEDS, 'development_evaluated': False, 'test_evaluated': False}, indent=2), encoding='utf-8')
    (out/'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    predictions = []
    for seed in SEEDS:
        for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
            fit, valid = {k:x.iloc[a] for k,x in data.items()}, {k:x.iloc[b] for k,x in data.items()}
            target, held = y.iloc[a], y.iloc[b]
            splits = list(StratifiedKFold(3, shuffle=True, random_state=seed+fold+1).split(np.zeros(len(a)), target))
            audit = {'fit_ids': list(map(int, target.index)), 'heldout_ids': list(map(int, held.index)),
                     'inner_ids': [{'fit': list(map(int, target.index[ia])), 'heldout': list(map(int, target.index[ib]))} for ia,ib in splits],
                     'candidates': {}, 'selected': {}}
            specifications, all_text = {}, []
            for family in ['logistic', 'linear']:
                key = 'previous_'+family
                records = grid(fit['mean'], target, splits, 'pca', [family], [0.], seed)
                for r in records:r['feature'] = 'mean'
                audit['candidates'][key] = records
                specifications[key] = select(records)
            for feature in ['mean', 'rich']:
                for scaling in ['standard', 'l2']:
                    key = feature+'_'+scaling
                    logging.info('Text refinement seed %s fold %s: %s', seed, fold, key)
                    records = grid(fit[feature], target, splits, scaling, ['logistic', 'linear', 'rbf'], [0., -.5, .5], seed)
                    for r in records:r['feature'] = feature
                    audit['candidates'][key] = records
                    specifications[key+'_default'], specifications[key+'_tuned'] = select(records, True), select(records)
                    all_text.extend(records)
            specifications['selected_text'] = select(all_text)
            branches = [select([r for r in audit['candidates']['mean_standard'] if r['family'] == 'logistic'], True)]
            for feature in ['audio', 'face']:
                records = grid(fit[feature], target, splits, 'base', ['logistic'], [0.], seed)
                for r in records:r['feature'] = feature
                audit['candidates'][feature] = records
                branches.append(select(records))
            fusion, fusion_records = fusion_search(branches, target)
            audit['candidates']['fusion'] = fusion_records
            audit['fusion_branches'] = branches
            audit['selected'] = {**specifications, 'late_fusion': fusion}
            audit['selected_strategy'] = 'selected_text' if specifications['selected_text']['inner_f1'] >= fusion['inner_f1'] else 'late_fusion'
            models = {}
            fitted = {}
            def trained(spec):
                key = json.dumps({k:spec[k] for k in ['feature', 'family', 'parameters', 'scaling']}, sort_keys=True)
                if key not in fitted:
                    fitted[key] = fit_candidate(fit[spec['feature']], target, spec, seed)
                return fitted[key]
            for name, spec in specifications.items():
                models[name] = {'kind': 'text', 'feature': spec['feature'], 'model': trained(spec), 'threshold': spec['threshold']}
            models['late_text_only'] = {'kind':'text', 'feature':'mean', 'model':trained(branches[0]), 'threshold':0.}
            models['late_fusion'] = {'kind':'fusion', 'models':[trained(s) for s in branches], 'weights':fusion['weights'], 'threshold':fusion['threshold']}
            models['selected_strategy'] = models[audit['selected_strategy']]
            for name, bundle in models.items():
                raw, scores = predict(bundle, valid)
                checkpoint = out/'models'/f'{name}_{seed}_{fold}.joblib'
                joblib.dump(bundle, checkpoint)
                np.testing.assert_array_equal(scores, predict(joblib.load(checkpoint), valid)[1])
                if not np.isfinite(scores).all():raise ValueError('Invalid heldout scores')
                predictions.extend({'condition':name, 'scenario':'clean', 'seed':seed, 'fold':fold,
                    'participant_id':int(pid), 'target':int(t), 'raw_score':float(r), 'threshold':bundle['threshold'],
                    'score':float(s), 'prediction':int(s>=0)} for pid,t,r,s in zip(held.index, held, raw, scores))
            for constant in [0,1]:
                predictions.extend({'condition':f'constant_{constant}', 'scenario':'clean', 'seed':seed, 'fold':fold,
                    'participant_id':int(pid), 'target':int(t), 'raw_score':float(constant), 'threshold':.5,
                    'score':constant-.5, 'prediction':constant} for pid,t in held.items())
            (out/f'fold_{seed}_{fold}.json').write_text(json.dumps(audit, indent=2, allow_nan=False), encoding='utf-8')
    pred = pd.DataFrame(predictions)
    for _,g in pred.groupby(['condition','seed']):
        if len(g)!=len(y) or not g.participant_id.is_unique or set(g.participant_id)!=set(y.index):raise ValueError('Incomplete OOF')
    pred.to_parquet(out/'predictions.parquet', index=False)
    summarize(pred, out)
    for name,digest in hashes.items():
        if sha256(project/name)!=digest:raise ValueError('Source changed during fitting')
    (out/'verification.json').write_text(json.dumps({'participants':len(y), 'positives':int(y.sum()),
        'prediction_rows':len(pred), 'model_hashes':{p.name:sha256(p) for p in (out/'models').iterdir()},
        'hashes_verified':True, 'test_evaluated':False}, indent=2), encoding='utf-8')
    return out


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    with threadpool_limits(limits=1):
        print(run(Path(__file__).resolve().parents[2]))
