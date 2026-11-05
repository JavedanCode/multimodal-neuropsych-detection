"""Exercise cache reuse across the real alignment/sorting path."""
import json
import numpy as np
import pandas as pd
from src.dataset import aligned_v2
from src.experiments.baselines import sha256


def test_sorted_alignment_preserves_cache_key_and_reuses_artifact(tmp_path,monkeypatch):
    project=tmp_path;processed=project/'data/processed';raw=project/'data/raw'
    emb=processed/'mpnet_v1/participant_cache';emb.mkdir(parents=True);raw.mkdir(parents=True)
    source=project/'src/dataset/aligned_v2.py';source.parent.mkdir(parents=True)
    for name in ['aligned_v2.py','contracts_v2.py','speech_embeddings.py']:
        (source.parent/name).write_text('fixture source')
    monkeypatch.setattr(aligned_v2,'__file__',str(source))
    (raw/'302_P.tar.gz').write_bytes(b'fixture archive')
    labels=pd.DataFrame({'PHQ_Binary':[0]},index=[302]);labels.to_parquet(processed/'mpnet_v1/labels.parquet')
    labels.assign(split='train').to_parquet(processed/'labels.parquet')
    np.savez(emb/'302.npz',embeddings=np.array([[20.]*768,[10.]*768]))
    prior={'participants':{'302':{'hash':sha256(emb/'302.npz'),'key':{'transcript_hash':'fixture'},'audit':{'retained':2}}}}
    (processed/'mpnet_v1/manifest.json').write_text(json.dumps(prior))
    (processed/'stage3_report.json').write_text(json.dumps({'schemas':{'audio':{'columns':['value']},'facial':{'columns':['value']}}}))
    data={'wave':np.ones(32000,dtype=np.float32)*.01,'wave_meta':{'duration_seconds':2.},
          'text':pd.DataFrame({'Start_Time':[1.,0.],'End_Time':[1.5,.5],'Text':['later','earlier']}),
          'audio':pd.DataFrame({'frameTime':[0.,1.,1.99],'value':[1.,2.,3.]}),
          'face':pd.DataFrame({'timestamp':[0.,1.,1.99],'value':[4.,5.,6.],'success':[1]*3,'confidence':[.9]*3})}
    monkeypatch.setattr(aligned_v2,'read',lambda path:(data,{'text':'fixture'}))
    out=aligned_v2.build(project)
    manifest=json.loads((out/'manifest.json').read_text());key=manifest['participants']['302']['key']
    assert isinstance(key,dict) and key['sources']==manifest['sources']
    with np.load(out/'participant_cache/302.npz') as z:
        np.testing.assert_array_equal(z['starts'],[0.,1.])
        np.testing.assert_array_equal(z['text'][:,0],[10.,20.])
        assert not z['quality'][:,9:11].any()
    before=(out/'participant_cache/302.npz').stat().st_mtime_ns
    def fail_read(path):raise AssertionError('Unchanged cached data must not reopen archive')
    monkeypatch.setattr(aligned_v2,'read',fail_read)
    aligned_v2.build(project)
    assert (out/'participant_cache/302.npz').stat().st_mtime_ns==before
