"""Independent saved-model, fold, and sklearn-metric checks for the follow-up."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from src.experiments.baselines import sha256
from src.experiments.text_temporal_study import load
from src.models.temporal_fusion import MODALITIES, TemporalFusion, perturb


def verify(project, out):
    torch.set_num_threads(1)
    protocol = json.loads((out/'protocol.json').read_text())
    verification = json.loads((out/'verification.json').read_text())
    for name, digest in protocol['hashes'].items():
        assert sha256(project/name) == digest, name
    for name, digest in verification['model_hashes'].items():
        assert sha256(out/'models'/name) == digest, name
    sets, values, y = load(project)
    pred = pd.read_parquet(out/'predictions.parquet')
    assert not pred.duplicated(['condition', 'scenario', 'seed', 'participant_id']).any()
    np.testing.assert_array_equal(pred.target, y.loc[pred.participant_id])
    assert np.array_equal(pred.prediction, (pred.score >= 0).astype(int))
    metric = pd.read_csv(out/'repeat_metrics.csv').set_index(['condition', 'scenario', 'seed'])
    summary = pd.read_csv(out/'summary.csv').set_index(['condition', 'scenario'])
    gates = pd.read_parquet(out/'gates.parquet') if protocol['mode'] == 'temporal' else None
    for key, g in pred.groupby(['condition', 'scenario', 'seed']):
        assert len(g) == len(y) and set(g.participant_id) == set(y.index)
        tn, fp, fn, tp = confusion_matrix(g.target, g.prediction, labels=[0, 1]).ravel()
        expected = {'f1': f1_score(g.target, g.prediction, zero_division=0),
                    'precision': precision_score(g.target, g.prediction, zero_division=0),
                    'recall': recall_score(g.target, g.prediction, zero_division=0),
                    'accuracy': accuracy_score(g.target, g.prediction),
                    'balanced_accuracy': balanced_accuracy_score(g.target, g.prediction),
                    'specificity': tn/(tn+fp), 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                    'mean_fold_auc': np.mean([roc_auc_score(f.target, f.score) for _, f in g.groupby('fold')])}
        for name, value in expected.items():
            np.testing.assert_allclose(metric.loc[key, name], value, atol=1e-12)
    for key, g in metric.reset_index().groupby(['condition', 'scenario']):
        for name in ['f1', 'precision', 'recall', 'specificity', 'accuracy', 'balanced_accuracy', 'mean_fold_auc']:
            np.testing.assert_allclose(summary.loc[key, name+'_mean'], g[name].mean(), atol=1e-12)
            np.testing.assert_allclose(summary.loc[key, name+'_sd'], g[name].std(ddof=0), atol=1e-12)
    restored = 0
    for seed in protocol['seeds']:
        for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
            audit = json.loads((out/f'fold_{seed}_{fold}.json').read_text())
            assert audit['fit_ids'] == list(y.index[a]) and audit['heldout_ids'] == list(y.index[b])
            held = y.iloc[b]
            if protocol['mode'] == 'text':
                inner = list(StratifiedKFold(3, shuffle=True, random_state=seed+fold+1).split(np.zeros(len(a)), y.iloc[a]))
                for (ia, ib), recorded in zip(inner, audit['inner_ids']):
                    assert recorded == {'fit': list(y.index[a][ia]), 'heldout': list(y.index[a][ib])}
                for name, x in sets.items():
                    for family in ['logistic', 'linear']:
                        condition = name+'_'+family
                        assert audit['selected'][condition] == max(audit['candidates'][condition], key=lambda r:r['selection_f1'])
                        model = joblib.load(out/'models'/f'{condition}_{seed}_{fold}.joblib')
                        scores = model.decision_function(x.iloc[b])
                        saved = pred[(pred.condition == condition)&(pred.seed == seed)&(pred.fold == fold)].set_index('participant_id').loc[held.index]
                        np.testing.assert_allclose(saved.score, scores, rtol=1e-10, atol=1e-10)
                        restored += 1
            else:
                scaler = joblib.load(out/'models'/f'scaler_{seed}_{fold}.joblib')
                arrays = [values[m][b] for m in MODALITIES]
                xs = scaler.transform(arrays, values['masks'][b])
                m, q = torch.tensor(values['masks'][b]), torch.tensor(values['quality'][b])
                for condition in audit['losses']:
                    model = TemporalFusion([x.shape[2] for x in arrays], temporal=condition != 'mean_plain', quality=condition == 'temporal_quality').eval()
                    model.load_state_dict(torch.load(out/'models'/f'{condition}_{seed}_{fold}.pt', weights_only=True))
                    names = [condition, condition+'_blind'] if condition == 'temporal_quality' else [condition]
                    for name in names:
                        saved = pred[(pred.condition == name)&(pred.seed == seed)&(pred.fold == fold)]
                        for scenario, g in saved.groupby('scenario'):
                            cx, cm, cq = perturb(xs, m, q, scenario, seed*100+fold)
                            with torch.inference_mode():
                                scores, weights = model(cx, cm, cq, blind=name.endswith('_blind'))
                            np.testing.assert_allclose(g.set_index('participant_id').loc[held.index].score, scores, atol=1e-8)
                            np.testing.assert_allclose(weights.sum(1), 1, atol=1e-6)
                            assert (weights[~cm.any(2)] == 0).all()
                            stored = gates[(gates.condition == name)&(gates.scenario == scenario)&(gates.seed == seed)&(gates.fold == fold)].set_index('participant_id').loc[held.index]
                            np.testing.assert_allclose(stored[list(MODALITIES)], weights, atol=1e-8)
                    restored += 1
    matched = 0
    if protocol['mode'] == 'text':
        previous = project/'experiments/runs/speech_20261029T201655576434Z/predictions.parquet'
        old = pd.read_parquet(previous)
        columns = ['condition', 'seed', 'fold', 'participant_id']
        old = old[old.condition.str.startswith(('base_concat_', 'base_audio_'))].set_index(columns)
        renamed = pred.copy()
        renamed['condition'] = renamed.condition.str.replace('audio_only_', 'base_audio_', regex=False)
        current = renamed[renamed.condition.str.startswith(('base_concat_', 'base_audio_'))].set_index(columns).loc[old.index]
        np.testing.assert_allclose(current.score, old.score, rtol=1e-8, atol=1e-8)
        np.testing.assert_array_equal(current.prediction, old.prediction)
        matched = len(old)
    result = {'independent_metric_rows': len(metric), 'prediction_rows': len(pred), 'restored_classifiers': restored,
              'matched_previous_control_predictions': matched, 'folds_and_hashes_verified': True}
    (out/'independent_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(verify(Path(__file__).resolve().parents[2], args.run))
