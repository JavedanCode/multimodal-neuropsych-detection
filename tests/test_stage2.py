import io
import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest

from src.dataset import inventory as inv
from src.dataset.features import audit_features, inspect_features


@pytest.mark.parametrize('ids,targets', [([1.5],[0]), (['bad'],[0]), ([None],[0]), ([0],[1]), ([-1],[1]), ([float('inf')],[1]), ([1],[2]), ([1],[None]), ([1],[0.5])])
def test_reject_invalid_labels(tmp_path, ids, targets):
    path = tmp_path / 'train_split.csv'
    pd.DataFrame({'Participant_ID': ids, 'PHQ_Binary': targets}).to_csv(path,index=False)
    with pytest.raises(ValueError):
        inv.load_split_labels(path)


def test_within_split_duplicates_are_not_cross_split():
    result = inv.check_label_consistency({'train': pd.DataFrame({'Participant_ID':[1,1]})})
    assert result['duplicate_ids_within_split'] == {'train':[1]}
    assert result['ids_in_multiple_splits'] == {}


def test_config_paths_and_invalid_exit(tmp_path):
    root = tmp_path / 'raw'
    root.mkdir()
    pd.DataFrame({'Participant_ID':[1],'PHQ_Binary':[2]}).to_csv(root/'train_split.csv',index=False)
    config = tmp_path/'config.yaml'
    config.write_text(json.dumps({'paths':{'data_root':str(root),'manifest':str(tmp_path/'m.json'),'audit_report':str(tmp_path/'a.md')}}))
    assert inv.main(['--config',str(config)]) == 1
    assert 'train_split.csv' in json.loads((tmp_path/'m.json').read_text())['cross_checks']['schema_errors']
    assert inv.main(['--config',str(config),'--out',str(root/'unsafe.json')]) == 2
    assert not (root/'unsafe.json').exists()


def make_archive(path):
    with tarfile.open(path,'w:gz') as tar:
        for name, content in {'1_P/features/1_OpenSMILE2.3.0_egemaps.csv': b'name;frameTime;Loudness\nx;0;1\n', '1_P/features/1_OpenFace2.1.0_Pose_gaze_AUs.csv': b'frame,timestamp,confidence,success,AU01_r\n1,0,0.9,1,2\n'}.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            tar.addfile(member,io.BytesIO(content))


def test_feature_headers_cache_and_read_only(tmp_path, monkeypatch):
    root=tmp_path/'raw'; root.mkdir()
    path=root/'1_P.tar.gz'; make_archive(path)
    before=path.read_bytes()
    output=tmp_path/'features.json'
    report=audit_features(root,output)
    assert report['summary']['archives_with_both_header_types']==1
    assert report['summary']['archives_with_errors']==0
    files=report['archives']['1_P.tar.gz']['files']
    assert files[0]['columns']==['name','frameTime','Loudness']
    assert files[1]['quality_columns']==['confidence','success']
    monkeypatch.setattr('src.dataset.features.inspect_features',lambda _: pytest.fail('cache not reused'))
    assert audit_features(root,output)['summary']==report['summary']
    assert path.read_bytes()==before
    assert list(root.iterdir())==[path]
    with pytest.raises(ValueError,match='outside raw'):
        audit_features(root,root/'bad.json')


def test_corrupt_archive(tmp_path):
    path=tmp_path/'1_P.tar.gz'; path.write_bytes(b'bad')
    assert inspect_features(path)['errors']


def test_feature_cache_invalidates_changed_archive(tmp_path, monkeypatch):
    root=tmp_path/'raw'; root.mkdir()
    path=root/'1_P.tar.gz'; make_archive(path)
    output=tmp_path/'features.json'
    audit_features(root,output)
    path.write_bytes(b'corrupted')
    report=audit_features(root,output)
    assert report['summary']['archives_with_errors']==1
    calls=[]
    monkeypatch.setattr('src.dataset.features.inspect_features', lambda p: calls.append(p) or {'files':[], 'errors':['still bad']})
    audit_features(root,output)
    assert calls == [path.resolve()]


def test_feature_unsafe_and_duplicate_header(tmp_path):
    path=tmp_path/'1_P.tar.gz'
    with tarfile.open(path,'w:gz') as tar:
        for name,data in {'../bad':b'x','1_P/features/1_OpenFace2.csv':b'frame,frame\n'}.items():
            member=tarfile.TarInfo(name); member.size=len(data)
            tar.addfile(member,io.BytesIO(data))
    report=inspect_features(path)
    assert len(report['errors'])==2


def test_invalid_config_returns_actionable_exit(tmp_path):
    assert inv.main(['--config',str(tmp_path/'missing.yaml')]) == 2
    path=tmp_path/'bad.yaml'; path.write_text('- list\n- not a mapping\n')
    assert inv.main(['--config',str(path)]) == 2


def test_cli_overrides_config_paths(tmp_path):
    config=tmp_path/'config.yaml'
    config.write_text(json.dumps({'paths':{'data_root':str(tmp_path/'absent')}}))
    root=tmp_path/'raw'; root.mkdir()
    for split,pid in [('train',1),('dev',2),('test',3)]:
        pd.DataFrame({'Participant_ID':[pid],'PHQ_Binary':[0]}).to_csv(root/f'{split}_split.csv',index=False)
        (root/f'{pid}_P.tar.gz').write_bytes(b'shallow scan only')
    assert inv.main(['--config',str(config),'--root',str(root),'--out',str(tmp_path/'m.json'),'--report',str(tmp_path/'a.md')]) == 0
    assert json.loads((tmp_path/'m.json').read_text())['totals']['n_splits'] == 3
