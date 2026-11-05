import numpy as np
from src.experiments.precision_series import choose, measures, decision


def test_precision_selection_rejects_recall_collapse_and_f1_loss():
    ref={'metrics':{'precision':.48,'recall':.74,'f1':.58}}
    collapse={'metrics':{'precision':1.,'recall':.1,'f1':.18}}
    loss={'metrics':{'precision':.9,'recall':.65,'f1':.5}}
    good={'metrics':{'precision':.55,'recall':.7,'f1':.61}}
    assert choose([ref,collapse,loss,good],ref) is good
    assert choose([collapse,loss],ref) is ref


def test_agreement_requires_both_models_and_applies_threshold():
    class Prep:
        def transform(self,x):return x
    class Model:
        def __init__(self,scores):self.scores=scores
        def decision_function(self,x):return np.array(self.scores)
    b={'preprocess':Prep(),'models':[Model([1,-1,1]),Model([-1,1,.2])],'threshold':.3}
    np.testing.assert_allclose(decision(b,None),[-1.3,-1.3,-.1])
    assert measures([1,0,1],[-1,-1,-1])['precision']==0
