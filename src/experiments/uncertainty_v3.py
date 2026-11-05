"""Conditional paired participant bootstrap; never treat repeats as new people."""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from src.experiments.baselines import sha256


def bootstrap_weights(y,domains,draws=2000,seed=20261104):
    y=np.asarray(y);domains=np.asarray(domains);rng=np.random.default_rng(seed)
    strata=np.array([f'{d}_{t}' for d,t in zip(domains,y)]);weights=np.zeros((draws,len(y)),int)
    for name in np.unique(strata):
        indices=np.flatnonzero(strata==name)
        samples=rng.choice(indices,size=(draws,len(indices)),replace=True)
        for i,row in enumerate(samples):weights[i]+=np.bincount(row,minlength=len(y))
    return weights


def sampled_metrics(y,prediction,weights):
    y=np.asarray(y)[:,None];p=np.asarray(prediction)
    tp=weights@((y==1)&(p==1));fp=weights@((y==0)&(p==1));fn=weights@((y==1)&(p==0))
    return {'precision':np.mean(tp/np.maximum(tp+fp,1),axis=1),
            'f1':np.mean(2*tp/np.maximum(2*tp+fp+fn,1),axis=1)}


def run(out):
    pred=pd.read_parquet(out/'predictions.parquet').query('scenario=="clean"')
    ids=sorted(pred.participant_id.unique());labels=pred.groupby('participant_id').target.first().loc[ids].to_numpy()
    weights=bootstrap_weights(labels,np.where(np.array(ids)>=600,'AI','WoZ'))
    values={}
    for condition,g in pred.groupby('condition'):
        matrix=g.pivot(index='participant_id',columns='seed',values='prediction').loc[ids].to_numpy()
        assert matrix.shape==(len(ids),3) and np.isin(matrix,[0,1]).all()
        values[condition]=sampled_metrics(labels,matrix,weights)
    intervals={}
    for name,value in values.items():
        if name=='reference':continue
        intervals[name]={metric:{'lower':float(np.quantile(v-values['reference'][metric],.025)),
                                 'upper':float(np.quantile(v-values['reference'][metric],.975))} for metric,v in value.items()}
    report={'draws':len(weights),'seed':20261104,'unique_participants':len(ids),'repeats_carried_together':3,
            'method':'Paired participant bootstrap stratified by domain and released label; percentile 95% intervals',
            'limitation':'Conditional on existing predictions; excludes refitting and adaptive search uncertainty; not independent validation',
            'prediction_hash':sha256(out/'predictions.parquet'),'source_hash':sha256(Path(__file__)),'differences':intervals}
    (out/'conditional_intervals.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':print(json.dumps(run(Path(sys.argv[1])),indent=2))
