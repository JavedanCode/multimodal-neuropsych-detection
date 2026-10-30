"""Matched training-only text, sequence, and reliability experiments."""
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.experiments.audio_study import SEEDS, estimator, select_inner, metric_row
from src.experiments.baselines import sha256, TARGET_POLICY
from src.models.text_preprocessing import TextPreprocessor
from src.models.temporal_fusion import MODALITIES, SequenceScaler, TemporalFusion, augment, perturb

SCENARIOS = ['clean', 'audio_noise', 'face_loss', 'missing_text', 'missing_audio', 'missing_face']


def load(project):
    root = project / 'data/processed'
    directory = root / 'text_temporal_v1'
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['partition'] != 'train' or manifest['source'] != sha256(project / 'src/dataset/text_temporal.py'):
        raise ValueError('Extractor provenance mismatch')
    for name, digest in manifest['output_hashes'].items():
        if sha256(directory / name) != digest:
            raise ValueError('Input changed')
    y = pd.read_parquet(directory / 'labels.parquet').PHQ_Binary
    official = pd.read_parquet(root / 'labels.parquet')
    previous = pd.read_parquet(root / 'segmentation_study_v1/labels.parquet').PHQ_Binary
    if not y.equals(previous) or not y.equals(official.loc[y.index, 'PHQ_Binary']) or not (official.loc[y.index, 'split'] == 'train').all():
        raise ValueError('Training cohort changed')
    with np.load(directory / 'sequences.npz') as data:
        values = {k: data[k] for k in data.files}
    np.testing.assert_array_equal(values.pop('ids'), y.index)
    for x in values.values():
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite features')
    text = pd.DataFrame(values['text_mean'], index=y.index, columns=[f'text__{i}' for i in range(384)])
    base = pd.read_parquet(root / 'features.parquet').loc[y.index]
    sets = {'base_concat': base, 'text': text, 'text_concat': pd.concat([text, base], axis=1),
            'audio_only': base.filter(like='audio__'), 'face_only': base.filter(like='facial__')}
    return sets, values, y


def summarize(pred, out):
    rows, domains = [], []
    for (condition, scenario, seed), g in pred.groupby(['condition', 'scenario', 'seed']):
        if not g.participant_id.is_unique:
            raise ValueError('Repeated participant in repeat')
        aucs = [roc_auc_score(f.target, f.score) for _, f in g.groupby('fold')]
        rows.append({'condition': condition, 'scenario': scenario, 'seed': seed,
                     **metric_row(g.target, g.prediction), 'mean_fold_auc': float(np.mean(aucs))})
        for domain, d in g.groupby(np.where(g.participant_id >= 600, 'AI', 'WoZ')):
            domains.append({'condition': condition, 'scenario': scenario, 'seed': seed, 'domain': domain,
                            'n': len(d), 'positives': int(d.target.sum()), **metric_row(d.target, d.prediction)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / 'repeat_metrics.csv', index=False)
    pd.DataFrame(domains).to_csv(out / 'domain_metrics.csv', index=False)
    columns = ['f1', 'precision', 'recall', 'specificity', 'accuracy', 'balanced_accuracy', 'mean_fold_auc']
    summary = metrics.groupby(['condition', 'scenario'])[columns].agg(['mean', lambda x: x.std(ddof=0)])
    summary.columns = [a + ('_mean' if b == 'mean' else '_sd') for a, b in summary.columns]
    summary.to_csv(out / 'summary.csv')
    return metrics


def save_predictions(records, condition, scenario, seed, fold, held, scores):
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite prediction')
    records.extend({'condition': condition, 'scenario': scenario, 'seed': seed, 'fold': fold,
                    'participant_id': int(pid), 'target': int(t), 'score': float(s), 'prediction': int(s >= 0)}
                   for pid, t, s in zip(held.index, held, scores))


def run_text(sets, y, out):
    records = []
    for seed in SEEDS:
        for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
            fit_y, held = y.iloc[a], y.iloc[b]
            inner = list(StratifiedKFold(3, shuffle=True, random_state=seed+fold+1).split(np.zeros(len(a)), fit_y))
            audit = {'fit_ids': list(map(int, fit_y.index)), 'heldout_ids': list(map(int, held.index)),
                     'inner_ids': [{'fit': list(map(int, fit_y.index[ia])), 'heldout': list(map(int, fit_y.index[ib]))} for ia, ib in inner],
                     'selected': {}, 'candidates': {}}
            for name, x in sets.items():
                logging.info('Text seed %s fold %s: %s', seed, fold, name)
                fit, valid = x.iloc[a], x.iloc[b]
                prepared = []
                for ia, ib in inner:
                    prep = TextPreprocessor().fit(fit.iloc[ia])
                    prepared.append((ia, ib, prep.transform(fit.iloc[ia]), prep.transform(fit.iloc[ib])))
                for family in ['logistic', 'linear']:
                    condition = name + '_' + family
                    selected, _, candidates = select_inner(fit, fit_y, family, prepared, seed, [0.])
                    audit['selected'][condition], audit['candidates'][condition] = selected, candidates
                    model = Pipeline([('preprocess', TextPreprocessor()), ('classifier', estimator(family, selected['parameters'], seed))]).fit(fit, fit_y)
                    scores = model.decision_function(valid)
                    checkpoint = out / 'models' / f'{condition}_{seed}_{fold}.joblib'
                    joblib.dump(model, checkpoint)
                    np.testing.assert_array_equal(scores, joblib.load(checkpoint).decision_function(valid))
                    save_predictions(records, condition, 'clean', seed, fold, held, scores)
            for constant in [0, 1]:
                save_predictions(records, f'constant_{constant}', 'clean', seed, fold, held, np.full(len(held), 2*constant-1))
            (out / f'fold_{seed}_{fold}.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return pd.DataFrame(records)


def run_sequences(values, y, out):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    arrays = [values[m] for m in MODALITIES]
    masks, quality = values['masks'], values['quality']
    records, gates = [], []
    dimensions = [x.shape[2] for x in arrays]
    for seed in SEEDS:
        for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
            fit_y, held = y.iloc[a], y.iloc[b]
            scaler = SequenceScaler().fit([x[a] for x in arrays], masks[a])
            fit_x = scaler.transform([x[a] for x in arrays], masks[a])
            valid_x = scaler.transform([x[b] for x in arrays], masks[b])
            fit_m, valid_m = torch.tensor(masks[a]), torch.tensor(masks[b])
            fit_q, valid_q = torch.tensor(quality[a]), torch.tensor(quality[b])
            target = torch.tensor(fit_y.to_numpy(), dtype=torch.float32)
            audit = {'fit_ids': list(map(int, fit_y.index)), 'heldout_ids': list(map(int, held.index)), 'epochs': 50, 'losses': {}}
            joblib.dump(scaler, out / 'models' / f'scaler_{seed}_{fold}.joblib')
            specifications = [('mean_plain', False, False, None), ('temporal_plain', True, False, None), ('temporal_quality', True, True, None)]
            specifications += [('available_without_'+m, True, False, m) for m in MODALITIES]
            for condition, temporal, aware, absent in specifications:
                logging.info('Temporal seed %s fold %s: %s', seed, fold, condition)
                torch.manual_seed(seed+fold)
                model = TemporalFusion(dimensions, temporal=temporal, quality=aware)
                optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.01)
                loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=(len(target)-target.sum())/target.sum())
                train_x, train_m, train_q = fit_x, fit_m, fit_q
                if absent:
                    train_x, train_m, train_q = perturb(fit_x, fit_m, fit_q, 'missing_'+absent, seed)
                losses = []
                for epoch in range(50):
                    model.train()
                    xs, m, q = augment(train_x, train_m, train_q, seed*10000+fold*100+epoch)
                    optimizer.zero_grad()
                    logits, _ = model(xs, m, q)
                    loss = loss_fn(logits, target)
                    if not torch.isfinite(loss):
                        raise ValueError('Nonfinite training loss')
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
                audit['losses'][condition] = losses
                model.eval()
                checkpoint = out / 'models' / f'{condition}_{seed}_{fold}.pt'
                torch.save(model.state_dict(), checkpoint)
                restored = TemporalFusion(dimensions, temporal=temporal, quality=aware).eval()
                restored.load_state_dict(torch.load(checkpoint, weights_only=True))
                with torch.inference_mode():
                    np.testing.assert_array_equal(model(valid_x, valid_m, valid_q)[0], restored(valid_x, valid_m, valid_q)[0])
                    for scenario in (['missing_'+absent] if absent else SCENARIOS):
                        xs, m, q = perturb(valid_x, valid_m, valid_q, scenario, seed*100+fold)
                        for blind in ([False, True] if aware else [False]):
                            name = condition + ('_blind' if blind else '')
                            scores, weights = model(xs, m, q, blind=blind)
                            save_predictions(records, name, scenario, seed, fold, held, scores.numpy())
                            gates.extend({'condition': name, 'scenario': scenario, 'seed': seed, 'fold': fold,
                                          'participant_id': int(pid), **dict(zip(MODALITIES, map(float, w)))}
                                         for pid, w in zip(held.index, weights.numpy()))
            (out / f'fold_{seed}_{fold}.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    pd.DataFrame(gates).to_parquet(out / 'gates.parquet', index=False)
    return pd.DataFrame(records)


def run(project, mode):
    sets, values, y = load(project)
    out = project / 'experiments/runs' / (mode + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    (out / 'models').mkdir()
    paths = ['docs/text_temporal_protocol.md', 'src/dataset/text_temporal.py', 'src/experiments/text_temporal_study.py',
             'src/experiments/compare_text_temporal.py',
             'src/models/text_preprocessing.py', 'src/models/temporal_fusion.py', 'src/experiments/audio_study.py',
             'src/dataset/preprocessing.py', 'src/experiments/metrics.py', 'src/experiments/baselines.py',
             'data/processed/text_temporal_v1/sequences.npz', 'data/processed/text_temporal_v1/labels.parquet',
             'data/processed/text_temporal_v1/manifest.json', 'data/processed/features.parquet', 'data/processed/labels.parquet']
    hashes = {p: sha256(project / p) for p in paths}
    (out / 'protocol.json').write_text(json.dumps({'hashes': hashes, 'target': TARGET_POLICY, 'mode': mode,
        'participants': list(map(int, y.index)), 'seeds': SEEDS, 'development_evaluated': False, 'test_evaluated': False}, indent=2), encoding='utf-8')
    (out / 'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    pred = run_text(sets, y, out) if mode == 'text' else run_sequences(values, y, out)
    for _, g in pred.groupby(['condition', 'scenario', 'seed']):
        if len(g) != len(y) or set(g.participant_id) != set(y.index):
            raise ValueError('Incomplete predictions')
    pred.to_parquet(out / 'predictions.parquet', index=False)
    summarize(pred, out)
    for p, digest in hashes.items():
        if sha256(project / p) != digest:
            raise ValueError('Source changed during fitting')
    (out / 'verification.json').write_text(json.dumps({'participants': len(y), 'positives': int(y.sum()),
        'rows': len(pred), 'hashes_verified': True, 'model_hashes': {p.name: sha256(p) for p in (out / 'models').iterdir()},
        'test_evaluated': False, 'development_evaluated': False}, indent=2), encoding='utf-8')
    return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['text', 'temporal'])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    with threadpool_limits(limits=1):
        print(run(Path(__file__).resolve().parents[2], args.mode))
