import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.experiments.text_refinement import probabilities, fusion_search, select, predict


def test_logistic_margin_probability_mapping():
    x=np.array([[-2.],[-1.],[1.],[2.]])
    model=LogisticRegression().fit(x,[0,0,1,1])
    np.testing.assert_allclose(probabilities(model.decision_function(x)),model.predict_proba(x)[:,1])


def test_fusion_can_reject_bad_modalities_with_stable_ties():
    y=pd.Series([0,0,1,1])
    branches=[{'inner_scores':[-4,-3,3,4]}, {'inner_scores':[4,3,-3,-4]}, {'inner_scores':[4,3,-3,-4]}]
    selected,records=fusion_search(branches,y)
    assert len(records)==12
    assert selected['weights']==[1.,0.,0.] and selected['threshold']==.5
    assert selected['inner_f1']==1


def test_threshold_selection_and_saved_prediction_margin():
    records=[{'threshold':0.,'inner_f1':.4},{'threshold':-.5,'inner_f1':.6}]
    assert select(records)['threshold']==-.5
    assert select(records,True)['threshold']==0
    x=pd.DataFrame({'x':[-2.,-1.,1.,2.]})
    model=LogisticRegression().fit(x,[0,0,1,1])
    raw,margin=predict({'kind':'text','feature':'mean','model':model,'threshold':-.5},{'mean':x})
    np.testing.assert_allclose(margin,raw+.5)
