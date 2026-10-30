"""Verify every cached input and replay one participant's local text encoding."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.experiments.baselines import sha256
from src.dataset.text_temporal import read_members, safe_rows, encode


def verify(project):
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    root = project/'data/processed/text_temporal_v1'
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['source'] == sha256(project/'src/dataset/text_temporal.py')
    for name, digest in manifest['output_hashes'].items():
        assert sha256(root/name) == digest
    for name, digest in manifest['models'].items():
        assert sha256(project/'models/pretrained/all-MiniLM-L6-v2'/name) == digest
    with np.load(root/'sequences.npz') as data:
        arrays = {k:data[k] for k in data.files}
    labels = pd.read_parquet(root/'labels.parquet')
    np.testing.assert_array_equal(arrays['ids'], labels.index)
    for i, pid in enumerate(arrays['ids']):
        saved = manifest['participants'][str(pid)]
        assert saved['key']['source'] in [manifest['source'], manifest['compatible_cache_source']]
        old_face = json.loads((project/'data/processed/representation_study_v1/participant_cache'/f'{pid}.json').read_text())
        old_audit = json.loads((project/'reports/data_audit/label_speech_cache'/f'{pid}.json').read_text())
        expected_quarantine = abs(old_face['data']['facial']['diagnostics']['frame_end']-old_audit['result']['audio_seconds']) > 1
        assert saved['audit'].get('facial_timing_quarantined', False) == expected_quarantine
        path = root/'participant_cache'/f'{pid}.npz'
        assert sha256(path) == saved['sha256']
        stat = (project/'data/raw'/f'{pid}_P.tar.gz').stat()
        assert stat.st_size == saved['key']['archive_size'] and stat.st_mtime_ns == saved['key']['mtime']
        with np.load(path) as cache:
            for key in cache.files:
                np.testing.assert_array_equal(cache[key], arrays[key][i])
        for branch, name in enumerate(['text', 'audio', 'face']):
            assert arrays[name].shape[1] == 32
            quarantine = branch == 2 and saved['audit'].get('facial_timing_quarantined', False)
            assert arrays['masks'][i, branch].any() != quarantine
            assert (arrays[name][i][~arrays['masks'][i, branch]] == 0).all()
    assert np.isfinite(arrays['text_mean']).all()
    assert ((arrays['quality'] >= 0)&(arrays['quality'] <= 1)).all()
    assert (arrays['quality'][~arrays['masks']] == 0).all()
    pid = int(arrays['ids'][0])
    tables, hashes = read_members(project/'data/raw'/f'{pid}_P.tar.gz')
    assert hashes == manifest['participants'][str(pid)]['member_hashes']
    audit = json.loads((project/'reports/data_audit/label_speech_cache'/f'{pid}.json').read_text())
    duration = audit['result']['audio_seconds']
    texts, times, _, _ = safe_rows(tables['text'], duration)
    model_dir = project/'models/pretrained/all-MiniLM-L6-v2'
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True, use_safetensors=True).eval().to('cuda')
    vectors = encode(texts, tokenizer, model)
    maximum = float(np.max(np.abs(vectors.mean(0)-arrays['text_mean'][0])))
    np.testing.assert_allclose(vectors.mean(0), arrays['text_mean'][0], atol=1e-6)
    bins = np.clip(np.floor(times/duration*32).astype(int), 0, 31)
    for b in range(32):
        if (bins == b).any():
            np.testing.assert_allclose(vectors[bins == b].astype(float).mean(0), arrays['text'][0, b], atol=1e-6)
    result = {'verified_caches': len(labels), 'model_files_verified': True,
              'replayed_participants': 1, 'text_mean_max_replay_difference': maximum,
              'audit_totals': {k:sum(r['audit'][k] for r in manifest['participants'].values())
                              for k in ['rows', 'invalid_or_empty', 'direct_disclosures', 'excluded_with_neighbors', 'retained', 'unknown_confidence']}}
    (root/'independent_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    print(verify(Path(__file__).resolve().parents[2]))
