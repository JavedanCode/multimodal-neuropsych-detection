import numpy as np
import pandas as pd
from src.models.undersampling import sample_positions


def test_sampling_retains_positives_and_domain_proportions():
    y=pd.Series([1]*5+[0]*15+[0]*5,index=list(range(100,120))+list(range(600,605)))
    for ratio in (1,2):
        a=sample_positions(y,ratio,42)
        assert len(a)==5*(1+ratio) and len(np.unique(a))==len(a)
        assert set(range(5)).issubset(a)
        assert abs(sum(y.index[a]>=600)-ratio*5*.25)<=1
        np.testing.assert_array_equal(a,sample_positions(y,ratio,42))


def test_sampling_never_uses_rows_outside_fit():
    y=pd.Series([1,1,0,0,0,0,0,0],index=[101,102,103,104,600,601,602,603])
    fit=y.drop([102,603])
    a=sample_positions(fit,2,7)
    assert set(fit.index[a]).isdisjoint({102,603})
    assert 101 in fit.index[a]


def test_classifier_samples_rows_of_preprocessed_dataframe():
    from src.experiments.undersampling import fit_models
    y=pd.Series([1,1,0,0,0,0],index=[100,101,102,103,600,601])
    x=pd.DataFrame({'component_a':[2,3,-2,-3,-4,-5],'component_b':[0,1,0,1,0,1]},index=y.index)
    models,ids=fit_models(x,y,'single_1',{'C':.1,'class_weight':None},42)
    assert len(models)==1 and len(ids[0])==4
    assert np.isfinite(models[0].decision_function(x)).all()
