"""Predeclared paired exploratory screens; never chooses on the historical test."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

MEASURES = ['f1', 'precision', 'recall', 'specificity', 'accuracy', 'balanced_accuracy', 'mean_fold_auc']


def paired(metrics, candidate, control, scenario='clean'):
    sub = metrics[metrics.scenario == scenario].set_index(['condition', 'seed'])
    a, b = sub.loc[candidate].sort_index(), sub.loc[control].sort_index()
    assert a.index.equals(b.index)
    differences = a[MEASURES]-b[MEASURES]
    means = differences.mean()
    passes = means.f1 >= .03 and means.balanced_accuracy >= 0 and means.mean_fold_auc >= 0 and means.recall >= -.05 and (differences.f1 > 0).sum() >= 2
    return {'candidate': candidate, 'control': control, 'scenario': scenario,
            'mean_differences': means.to_dict(), 'f1_positive_repeats': int((differences.f1 > 0).sum()),
            'passes_clean_screen': bool(passes), 'repeat_differences': differences.reset_index().to_dict('records')}


def compare(text_run, temporal_run):
    text = pd.read_csv(text_run/'repeat_metrics.csv')
    temporal = pd.read_csv(temporal_run/'repeat_metrics.csv')
    clean = []
    for family in ['logistic', 'linear']:
        for candidate in ['text', 'text_concat']:
            clean.append(paired(text, candidate+'_'+family, 'base_concat_'+family))
        clean.append(paired(text, 'text_concat_'+family, 'text_'+family))
    clean.append(paired(temporal, 'temporal_plain', 'mean_plain'))
    quality = [paired(temporal, 'temporal_quality', 'temporal_plain', s)
               for s in ['clean', 'audio_noise', 'face_loss', 'missing_text', 'missing_audio', 'missing_face']]
    blind = [paired(temporal, 'temporal_quality', 'temporal_quality_blind', s)
             for s in ['clean', 'audio_noise', 'face_loss', 'missing_text', 'missing_audio', 'missing_face']]
    missing = [paired(temporal, 'temporal_quality', 'available_without_'+m, 'missing_'+m) for m in ['text', 'audio', 'face']]
    corrupt_gain = np.mean([r['mean_differences']['f1'] for r in quality if r['scenario'] != 'clean'])
    result = {'clean_screens': clean, 'quality_vs_plain': quality, 'quality_vs_blind': blind, 'missing_vs_available_training': missing,
              'quality_target': {'clean_f1_difference': quality[0]['mean_differences']['f1'],
                                 'mean_stressed_f1_difference': float(corrupt_gain),
                                 'passes': bool(quality[0]['mean_differences']['f1'] >= -.02 and corrupt_gain >= .05)},
              'interpretation': 'Exploratory repeated training CV; overlapping repeats are not independent replication. No new holdout evaluation.'}
    (temporal_run/'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=Path)
    parser.add_argument('temporal', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.text, args.temporal), indent=2))
