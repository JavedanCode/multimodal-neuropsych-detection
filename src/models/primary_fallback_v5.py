"""Audio/facial primary prediction with observable-quality-only fallback routing."""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.models.bounded_v3 import eligible_quality

SCENARIOS=('clean','missing_audio','missing_face','missing_both')


def availability(data,scenario,quality):
    if scenario not in SCENARIOS:raise ValueError('Unknown scenario')
    if quality:return eligible_quality(data,scenario)
    a=np.array(data['available'][:,1:],dtype=bool,copy=True)
    if scenario in ['missing_audio','missing_both']:a[:,0]=False
    if scenario in ['missing_face','missing_both']:a[:,1]=False
    return a


def design(scores,data,quality,scenario='clean'):
    a=availability(data,scenario,quality)
    # Deliberately ignore text margin at column zero and all text-quality fields.
    s=np.where(a,np.asarray(scores)[:,1:3],0.)
    parts=[s,a.astype(float)]
    if quality:
        parts.extend([s[:,0:1]*data['quality'][:,:6],s[:,1:2]*data['quality'][:,6:9]])
    return np.concatenate(parts,axis=1)


def fit_meta(scores,data,y,quality,c,seed):
    z=np.concatenate([design(scores,data,quality,s) for s in SCENARIOS[:3]])
    return Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=c,class_weight='balanced',solver='liblinear',max_iter=5000,random_state=seed))]).fit(z,np.tile(y,3),model__sample_weight=np.full(3*len(y),1/3))


def from_scores(meta,scores,data,quality,fallback,scenario='clean'):
    route=~availability(data,scenario,quality).any(1) if fallback else np.zeros(len(scores),bool)
    if np.any(route&~data['available'][:,0].astype(bool)):raise ValueError('Text unavailable for required fallback')
    primary=meta.decision_function(design(scores,data,quality,scenario))
    return np.where(route,scores[:,0],primary),route


def predict(bundle,data,scenario='clean'):
    n=len(data['available']);scores=np.zeros((n,3))
    for i,name in enumerate(['audio','face'],1):scores[:,i]=bundle['base'][name].decision_function(data[name])
    route=~availability(data,scenario,bundle['quality']).any(1) if bundle['fallback'] else np.zeros(n,bool)
    if np.any(route&~data['available'][:,0].astype(bool)):raise ValueError('Text unavailable for required fallback')
    # Text features are read only for routed participants at inference.
    if route.any():scores[route,0]=bundle['base']['text'].decision_function(data['text'].iloc[np.flatnonzero(route)])
    return from_scores(bundle['meta'],scores,data,bundle['quality'],bundle['fallback'],scenario)
