"""Fold-fitted compression for supplied deep features, separate from face features."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.dataset.preprocessing import ParticipantPreprocessor


class RepresentationPreprocessor(TransformerMixin,BaseEstimator):
    def __init__(self,n_components=32):
        self.n_components=n_components

    def fit(self,X,y=None):
        ParticipantPreprocessor._validate(X)
        self.input_columns_=list(X)
        self.dense_columns_=[c for c in X if c.startswith('audio__dense_')]
        self.other_columns_=[c for c in X if c not in self.dense_columns_]
        self.columns_=[]
        if self.dense_columns_:
            clean=X[self.dense_columns_].replace([np.inf,-np.inf],np.nan)
            self.imputer_=SimpleImputer(strategy='median',keep_empty_features=True)
            v=self.imputer_.fit_transform(clean)
            self.variance_=VarianceThreshold();v=self.variance_.fit_transform(v)
            self.scaler_=StandardScaler();v=self.scaler_.fit_transform(v)
            components=min(self.n_components,len(X)-1,v.shape[1])
            if components<1:raise ValueError('Too few fitting participants for PCA')
            self.pca_=PCA(n_components=components,svd_solver='randomized',random_state=42,whiten=True).fit(v)
            self.columns_ += [f'audio__dense_pc{i}' for i in range(components)]
        if self.other_columns_:
            self.other_=ParticipantPreprocessor().fit(X[self.other_columns_])
            self.columns_ += self.other_.columns_
        if not self.columns_:raise ValueError('Empty feature set')
        return self

    def transform(self,X):
        check_is_fitted(self,'columns_');ParticipantPreprocessor._validate(X)
        if list(X)!=self.input_columns_:raise ValueError('Feature schema/order differs')
        values=[]
        if self.dense_columns_:
            v=self.imputer_.transform(X[self.dense_columns_].replace([np.inf,-np.inf],np.nan))
            v=self.scaler_.transform(self.variance_.transform(v))
            values.append(self.pca_.transform(v))
        if self.other_columns_:values.append(self.other_.transform(X[self.other_columns_]).to_numpy())
        result=np.concatenate(values,axis=1)
        if not np.isfinite(result).all():raise ValueError('Nonfinite projected features')
        return pd.DataFrame(result,index=X.index,columns=self.columns_)
