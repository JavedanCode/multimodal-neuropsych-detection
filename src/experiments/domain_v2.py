"""Explicit label-and-domain stratification for prospective training evaluations."""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from src.dataset.contracts_v2 import domain


def splits(y,n_splits,seed):
    if not y.index.is_unique or not set(y.unique())<={0,1}:raise ValueError('Unique people and binary labels required')
    strata=np.array([f'{d}_{int(t)}' for d,t in zip(domain(y.index),y)])
    _,counts=np.unique(strata,return_counts=True)
    if counts.min()<n_splits:raise ValueError('Insufficient people for joint stratification')
    return list(StratifiedKFold(n_splits,shuffle=True,random_state=seed).split(np.zeros(len(y)),strata))


def transfer(y):
    d=domain(y.index);a=np.flatnonzero(d=='WoZ');b=np.flatnonzero(d=='AI')
    if not len(a) or not len(b):raise ValueError('Both interview domains required')
    return a,b
