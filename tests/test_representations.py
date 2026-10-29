import io
import tarfile

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.dataset.representations import facial_summary,dense_summary,extract
from src.dataset.preprocessing import ParticipantPreprocessor
from src.models.representation_preprocessing import RepresentationPreprocessor
from src.experiments.representation_study import prepare_inner,fit_outer


def test_facial_changes_do_not_bridge_failed_frames_or_time_gaps():
    frame=pd.DataFrame({'timestamp':[0,.04,.08,.12,1.2],
        'confidence':[1,1,.2,1,1],'success':[1]*5,'a':[0,2,100,6,40]})
    values,diagnostics=facial_summary(frame,['a'])
    assert diagnostics['retained_fraction']==.8
    assert values['facial__a__mean']==12
    assert values['facial__a__std']==pytest.approx(np.std([0,2,6,40]))
    assert values['facial__a__adjacent_change']==2
    assert values['facial__a__window_mean_std']==0
    frame.loc[0,'timestamp']=10
    with pytest.raises(ValueError,match='timeline'):facial_summary(frame,['a'])


def test_dense_summary_keeps_zero_and_ignores_nonfinite():
    frame=pd.DataFrame({'name':['x']*3,'timeStamp':[0,1,2],'neuron_0':[0,2,4],'neuron_1':[np.inf,np.nan,3]})
    values,diagnostics=dense_summary(frame,2)
    assert values['audio__dense_neuron_0__mean']==2
    assert values['audio__dense_neuron_1__mean']==3
    assert values['audio__dense_neuron_1__std']==0
    assert diagnostics['invalid_fraction']==pytest.approx(1/3)
    with pytest.raises(ValueError,match='schema'):dense_summary(frame,3)


def test_no_dense_branch_preserves_original_preprocessing():
    x=pd.DataFrame({'audio__a':[0.,1.,2.],'facial__b':[3.,1.,0.],'facial__constant':[1.,1.,1.]})
    a=RepresentationPreprocessor().fit_transform(x)
    b=ParticipantPreprocessor().fit_transform(x)
    pd.testing.assert_frame_equal(a,b)


def test_pca_and_other_preprocessing_fit_only_inner_training(monkeypatch):
    rng=np.random.default_rng(4)
    x=pd.DataFrame(rng.normal(size=(40,35)),columns=[f'audio__dense_n{i}' for i in range(34)]+['facial__a'])
    y=pd.Series([0,1]*20)
    splits=list(StratifiedKFold(3,shuffle=True,random_state=42).split(x,y))
    fitted=[];original=RepresentationPreprocessor.fit
    def track(self,values,y=None):
        fitted.append(list(values.index));return original(self,values,y)
    monkeypatch.setattr(RepresentationPreprocessor,'fit',track)
    prepare_inner(x,y,splits)
    assert fitted==[list(x.index[a]) for a,b in splits]
    selection={'parameters':{'C':.1,'class_weight':'balanced'}}
    valid=x.iloc[:4].copy();valid.index=range(100,104)
    a,_=fit_outer(x,y,valid,'logistic',selection,42)
    b,_=fit_outer(x,y,valid*100,'logistic',selection,42)
    np.testing.assert_array_equal(a['preprocess'].pca_.components_,b['preprocess'].pca_.components_)
    np.testing.assert_array_equal(a['classifier'].coef_,b['classifier'].coef_)
    assert a['preprocess'].pca_.n_components_==32
    with pytest.raises(ValueError,match='schema'):a['preprocess'].transform(valid[valid.columns[::-1]])


def test_archive_extraction_uses_deep_audio_csv_not_visual_mat(tmp_path):
    path=tmp_path/'300_P.tar.gz'
    dense=pd.DataFrame({'name':['300_AUDIO.wav']*2,'timeStamp':[0,1],**{f'neuron_{i}':[i,i+2] for i in range(1920)}})
    face=pd.DataFrame({'timestamp':[0,.04],'confidence':[1,1],'success':[1,1],'a':[0,2]})
    with tarfile.open(path,'w:gz') as archive:
        for name,frame in [('300_densenet201.csv',dense),('300_OpenFace2.1.0_Pose_gaze_AUs.csv',face)]:
            raw=frame.to_csv(index=False).encode();info=tarfile.TarInfo('300_P/features/'+name);info.size=len(raw)
            archive.addfile(info,io.BytesIO(raw))
    result=extract(path,['a'])
    assert result['dense']['features']['audio__dense_neuron_0__mean']==1
    assert result['facial']['features']['facial__a__mean']==1
    assert list(tmp_path.iterdir())==[path]
