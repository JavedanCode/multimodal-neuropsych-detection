"""Independent fixed-prediction metric and artifact checks for research runs."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score,balanced_accuracy_score,confusion_matrix,roc_auc_score
from src.experiments.baselines import sha256


def verify_outputs(project,out,y):
    protocol=json.loads((out/'protocol.json').read_text())
    completed=json.loads((out/'verification.json').read_text())
    for name,digest in protocol['hashes'].items():assert sha256(project/name)==digest,name
    for name,digest in completed['model_hashes'].items():assert sha256(out/'models'/name)==digest,name
    pred=pd.read_parquet(out/'predictions.parquet')
    metric=pd.read_csv(out/'repeat_metrics.csv').set_index(['condition','seed'])
    summary=pd.read_csv(out/'summary.csv').set_index('condition')
    assert not pred.duplicated(['condition','seed','participant_id']).any()
    np.testing.assert_array_equal(pred.target,y.loc[pred.participant_id])
    np.testing.assert_array_equal(pred.prediction,(pred.score>=0).astype(int))
    if 'raw_score' in pred:np.testing.assert_allclose(pred.score,pred.raw_score-pred.threshold,atol=1e-12)
    measures=['f1','precision','recall','specificity','accuracy','balanced_accuracy','mean_fold_auc']
    for key,g in pred.groupby(['condition','seed']):
        assert len(g)==len(y) and set(g.participant_id)==set(y.index)
        tn,fp,fn,tp=confusion_matrix(g.target,g.prediction,labels=[0,1]).ravel()
        expected=[f1_score(g.target,g.prediction,zero_division=0),precision_score(g.target,g.prediction,zero_division=0),
                  recall_score(g.target,g.prediction,zero_division=0),tn/(tn+fp),accuracy_score(g.target,g.prediction),
                  balanced_accuracy_score(g.target,g.prediction),np.mean([roc_auc_score(d.target,d.score) for _,d in g.groupby('fold')])]
        for name,value in zip(measures,expected):np.testing.assert_allclose(metric.loc[key,name],value,atol=1e-12)
    for condition,g in metric.reset_index().groupby('condition'):
        for name in measures:
            np.testing.assert_allclose(summary.loc[condition,name+'_mean'],g[name].mean(),atol=1e-12)
            np.testing.assert_allclose(summary.loc[condition,name+'_sd'],g[name].std(ddof=0),atol=1e-12)
    return protocol,pred,len(metric)
