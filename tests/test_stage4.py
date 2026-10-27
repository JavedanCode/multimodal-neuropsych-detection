import numpy as np
import pandas as pd
import pytest
from src.dataset.preprocessing import ParticipantPreprocessor
from src.experiments.baselines import fit_baseline, load_development_inputs, majority_baseline
from src.experiments.metrics import binary_metrics


def data():
    rng=np.random.default_rng(13)
    X=pd.DataFrame(rng.normal(size=(30,4)),index=pd.Index(range(1,31),name='Participant_ID'),columns=['audio__a','audio__b','facial__c','facial__d'])
    y=pd.Series(np.tile([0,1],15),index=X.index)
    return X.iloc[:24],y.iloc[:24],X.iloc[24:],y.iloc[24:]


def test_metrics_known_values_and_single_class():
    m=binary_metrics([0,0,1,1],[.1,.8,.4,.9])
    assert m['precision']==m['recall']==m['f1']==.5
    assert m['confusion_matrix']==[[1,1],[1,1]]
    assert m['roc_auc']==.75
    assert binary_metrics([0,0],[.1,.1])['roc_auc'] is None
    assert binary_metrics([0,1],[0,0])['f1']==0
    with pytest.raises(ValueError):binary_metrics([0,1],[np.nan,.2])
    with pytest.raises(ValueError):binary_metrics([0,2],[.1,.2])


def test_fold_preprocessing_and_development_exclusion(monkeypatch):
    train,y,dev,dy=data()
    observed=[]
    original=ParticipantPreprocessor.fit
    def track(self,X,y=None):
        observed.append(set(X.index))
        return original(self,X,y)
    monkeypatch.setattr(ParticipantPreprocessor,'fit',track)
    model,result,predictions=fit_baseline(train,y,dev,dy,'concatenation',42,3,[.1,1.],[None])
    assert len(observed)==7  # two candidates times three folds plus refit
    assert all(ids.issubset(set(train.index)) and not ids.intersection(dev.index) for ids in observed)
    assert sum(len(ids)==len(train) for ids in observed)==1
    assert predictions.index.equals(dev.index)
    assert result['evaluation_partition']=='dev'
    for fold in result['cv_participants']:
        assert not set(fold['train']).intersection(fold['validation'])
        assert set(fold['train'])|set(fold['validation'])==set(train.index)


def test_dev_labels_do_not_select_parameters_and_reproducible():
    train,y,dev,dy=data()
    first,r1,p1=fit_baseline(train,y,dev,dy,'audio',42,3,[.1,1.],[None,'balanced'])
    second,r2,p2=fit_baseline(train,y,dev,1-dy,'audio',42,3,[.1,1.],[None,'balanced'])
    assert r1['selected_parameters']==r2['selected_parameters']
    assert r1['cv_candidates']==r2['cv_candidates']
    np.testing.assert_array_equal(first.named_steps['classifier'].coef_,second.named_steps['classifier'].coef_)
    np.testing.assert_array_equal(p1.probability_class_1,p2.probability_class_1)
    assert r1['original_feature_count']==2
    assert all(c.startswith('audio__') for c in r1['retained_feature_names'])


def test_overlap_and_alignment_rejected():
    train,y,dev,dy=data()
    with pytest.raises(ValueError,match='overlap'):
        fit_baseline(train,y,train.iloc[:6],y.iloc[:6],'audio',42,3,[1.],[None])
    with pytest.raises(ValueError,match='aligned'):
        fit_baseline(train,y.iloc[::-1],dev,dy,'audio',42,3,[1.],[None])
    with pytest.raises(ValueError,match='Insufficient'):
        fit_baseline(train,y,dev,dy,'audio',42,13,[1.],[None])


def test_loader_ignores_test_values_and_normalized_files(tmp_path):
    train,y,dev,dy=data()
    X=pd.concat([train,dev]);labels=pd.DataFrame({'PHQ_Binary':pd.concat([y,dy]),'split':['train']*24+['dev']*4+['test']*2},index=X.index)
    X.loc[[29,30]]=np.nan
    labels.loc[[29,30],'PHQ_Binary']=999  # must never enter development experiment
    X.to_parquet(tmp_path/'features.parquet');labels.to_parquet(tmp_path/'labels.parquet')
    loaded=load_development_inputs(tmp_path)
    assert len(loaded[0])==24 and len(loaded[2])==4
    assert 999 not in loaded[1].values and 999 not in loaded[3].values
    assert not loaded[2].isna().any().any()


def test_majority_is_chosen_from_train():
    result,pred=majority_baseline(pd.Series([0,0,0,1]),pd.Series([1,1,1,0]))
    assert result['majority_class']==0
    assert pred.prediction.eq(0).all()
    assert result['dev_metrics']['accuracy']==.25
    assert result['dev_metrics']['f1']==0
