import io
import json
import tarfile
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from src.dataset.tables import aggregate, build
from src.dataset.preprocessing import ParticipantPreprocessor
from src.dataset.features import audit_features


def test_chunk_moments_missing_and_quality():
    data=b'name;frameTime;x;y\na;0;1;NaN\na;1;3;inf\na;2;5;7\n'
    results=[aggregate(io.BytesIO(data),['x','y'],'audio',chunk_rows=n) for n in (1,2,100)]
    for f,q,d in results:
        assert f['audio__x__mean']==3
        assert f['audio__x__std']==pytest.approx(np.std([1,3,5]))
        assert f['audio__y__mean']==7
        assert q['invalid_fraction']==pytest.approx(2/6)
        assert d['rows']==3


def test_face_filter_and_empty():
    data=b'frame,timestamp,confidence,success,AU01_r\n1,0,0.9,1,2\n2,1,0.2,1,100\n3,2,0.99,0,100\n4,3,2,1,100\n'
    f,q,d=aggregate(io.BytesIO(data),['AU01_r'],'facial',chunk_rows=1)
    assert f['facial__AU01_r__mean']==2
    assert q['retained_fraction']==0.25
    assert q['confidence_valid_fraction']==0.75
    f,q,d=aggregate(io.BytesIO(b'name;frameTime;x\n'),['x'],'audio')
    assert f['audio__x__mean'] is None
    assert q['available']==0


def test_training_only_preprocessing_and_clone():
    train=pd.DataFrame({'audio__x':[1.,3.,np.nan], 'audio__dup':[1.,3.,np.nan], 'facial__constant':[2.,2.,2.], 'facial__empty':[np.nan]*3})
    p=ParticipantPreprocessor().fit(train)
    assert p.columns_==['audio__x']
    assert p.medians_['audio__x']==2
    before=p.means_.copy()
    heldout=train.copy(); heldout['audio__x']=[1e9,np.nan,-1e9]
    out=p.transform(heldout)
    assert out.iloc[1,0]==0
    pd.testing.assert_series_equal(p.means_,before)
    assert not hasattr(clone(p),'columns_')
    fold=clone(p).fit(train.iloc[:2])
    assert fold.medians_['audio__x']==2
    with pytest.raises(ValueError): p.transform(train[train.columns[::-1]])
    with pytest.raises(ValueError): ParticipantPreprocessor().fit(pd.DataFrame({'PHQ_Binary':[0,1]}))
    with pytest.raises(ValueError): ParticipantPreprocessor().fit(pd.DataFrame({'audio__PHQ_Score':[0,1]}))


def fixture_dataset(root):
    root.mkdir()
    for pid in range(1,5):
        with tarfile.open(root/f'{pid}_P.tar.gz','w:gz') as tar:
            for suffix,data in [('OpenSMILE2.3.0_egemaps.csv',f'name;frameTime;x\na;0;{pid}\na;1;{pid+1}\n'),('OpenFace2.1.0_Pose_gaze_AUs.csv',f'frame,timestamp,confidence,success,AU01_r\n1,0,0.9,1,{pid*2}\n2,1,0.95,1,{pid*2+1}\n')]:
                payload=data.encode(); info=tarfile.TarInfo(f'{pid}_P/features/{pid}_{suffix}'); info.size=len(payload); tar.addfile(info,io.BytesIO(payload))
    for split,ids in [('train',[1,2]),('dev',[3]),('test',[4])]:
        pd.DataFrame({'Participant_ID':ids,'PHQ_Binary':[i%2 for i in ids]}).to_csv(root/f'{split}_split.csv',index=False)


def test_full_build_alignment_cache_and_leakage(tmp_path,monkeypatch):
    root=tmp_path/'raw';fixture_dataset(root)
    inventory=tmp_path/'inventory.json';audit_features(root,inventory)
    output=tmp_path/'processed'
    settings={'chunk_rows':1,'confidence_threshold':0.8}
    report=build(root,inventory,output,settings)
    assert report['participants']==4
    features=pd.read_parquet(output/'features.parquet')
    assert features.loc[3,'audio__x__mean']==3.5
    state=json.loads((output/'preprocessing.json').read_text())
    assert state['fit_participant_ids']==[1,2]
    assert state['means']['audio__x__mean']==2
    restored=ParticipantPreprocessor.from_state(state)
    expected=pd.read_parquet(output/'train_normalized.parquet')
    pd.testing.assert_frame_equal(restored.transform(features.loc[[1,2]]),expected)
    assert list(pd.read_parquet(output/'test_normalized.parquet').index)==[4]
    monkeypatch.setattr('src.dataset.tables.participant',lambda *a:pytest.fail('cache miss'))
    assert build(root,inventory,output,settings)==report
    pd.DataFrame({'Participant_ID':[2],'PHQ_Binary':[0]}).to_csv(root/'dev_split.csv',index=False)
    with pytest.raises(ValueError,match='overlap'):build(root,inventory,output,settings)


def test_fold_fitting_excludes_validation_extremes():
    raw=pd.DataFrame({'audio__x':[1.,3.,1000.], 'facial__x':[2.,6.,2000.]})
    fold=ParticipantPreprocessor().fit(raw.iloc[:2])
    assert fold.means_['audio__x']==2
    assert fold.scales_['audio__x']==1
    assert fold.transform(raw.iloc[2:]).iloc[0,0]==998
    assert fold.means_['audio__x']==2


def test_unusable_face_kept_missing_not_fabricated_zero():
    data=b'frame,timestamp,confidence,success,AU01_r\n1,0,0.1,1,22\n2,1,0.9,0,30\n'
    f,q,d=aggregate(io.BytesIO(data),['AU01_r'],'facial')
    assert f['facial__AU01_r__mean'] is None
    assert f['facial__AU01_r__std'] is None
    assert q['available']==0
    assert d['retained_rows']==0


def test_builder_rejects_raw_output_and_stale_inventory(tmp_path):
    root=tmp_path/'raw';fixture_dataset(root)
    inventory=tmp_path/'inventory.json';audit_features(root,inventory)
    settings={'chunk_rows':1,'confidence_threshold':0.8}
    with pytest.raises(ValueError,match='outside raw'):build(root,inventory,root/'out',settings)
    with (root/'1_P.tar.gz').open('ab') as f: f.write(b'x')
    with pytest.raises(ValueError,match='Stale'):build(root,inventory,tmp_path/'out',settings)
