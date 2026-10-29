import numpy as np
import pandas as pd
import pytest
import io
import json
import tarfile

from src.dataset.segmented_audio import moments, extract
from src.experiments.segmentation_study import load_inputs
from src.experiments.baselines import sha256


def test_coverage_preserves_duplicates_zeros_and_half_open_bounds():
    frame=pd.DataFrame({'frameTime':[0.,1.,1.,2.,3.], 'a':[100.,0.,2.,4.,200.]})
    result,coverage=moments(frame,['a'],[[1.,3.]])
    assert coverage==.6
    assert result['audio__a__mean']==2
    assert result['audio__a__std']==pytest.approx(np.std([0,2,4]))


def test_coverage_nonfinite_values_and_empty_selection():
    frame=pd.DataFrame({'frameTime':[0.,1.,2.], 'a':[np.nan,np.inf,4.]})
    result,_=moments(frame,['a'],[[0.,3.]])
    assert result['audio__a__mean']==4 and result['audio__a__std']==0
    with pytest.raises(ValueError,match='No transcript'):
        moments(frame,['a'],[[5.,6.]])
    with pytest.raises(ValueError,match='timeline'):
        moments(frame.assign(frameTime=[0.,2.,1.]),['a'],[[0.,3.]])


def test_archive_filters_actual_times_and_does_not_export_transcripts(tmp_path):
    path=tmp_path/'300_P.tar.gz'
    files={'300_Transcript.csv':b'Start_Time,End_Time,Text\n1,3,private words\n',
           '300_OpenSMILE2.3.0_egemaps.csv':b'frameTime;a\n0;99\n1;2\n2;4\n3;99\n'}
    with tarfile.open(path,'w:gz') as archive:
        for name,raw in files.items():
            info=tarfile.TarInfo('300_P/'+name);info.size=len(raw)
            archive.addfile(info,io.BytesIO(raw))
    result=extract(path,['a'],4.)
    assert result['features']['audio__a__mean']==3
    assert result['coverage']==.5
    assert 'private' not in json.dumps(result)
    assert list(tmp_path.iterdir())==[path]


def test_loader_rejects_holdout_and_modified_labels(tmp_path):
    root=tmp_path/'data/processed';directory=root/'segmentation_study_v1';directory.mkdir(parents=True)
    idx=pd.Index([300,301,600],name='Participant_ID')
    official=pd.DataFrame({'split':['train','train','test'],'PHQ_Binary':[0,1,1]},index=idx)
    official.to_parquet(root/'labels.parquet')
    x=pd.DataFrame({'audio__a__mean':[1.,2.,3.]},index=idx)
    x.assign(facial__a__mean=[2.,3.,4.]).to_parquet(root/'features.parquet')
    def save(ids,targets):
        x.loc[ids].to_parquet(directory/'features.parquet')
        pd.DataFrame({'PHQ_Binary':targets},index=ids).to_parquet(directory/'labels.parquet')
        (directory/'manifest.json').write_text(json.dumps({'partition':'train','sources':{},'excluded_ids':[],
            'output_hashes':{n:sha256(directory/n) for n in ['features.parquet','labels.parquet']}}))
    save(idx,[0,1,1])
    with pytest.raises(ValueError,match='training subset'):load_inputs(tmp_path)
    save(idx[:2],[1,0])
    with pytest.raises(ValueError,match='training subset'):load_inputs(tmp_path)
    save(idx[:2],[0,1])
    sets,y,_=load_inputs(tmp_path)
    assert len(sets)==5 and list(y.index)==[300,301]
