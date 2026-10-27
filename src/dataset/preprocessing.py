"""Training-fitted preprocessing reusable inside each cross-validation fold."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class ParticipantPreprocessor(TransformerMixin, BaseEstimator):
    """Drop train-empty/constant/duplicate columns, median-impute and standardize.

    Input is a descriptor-only DataFrame. Call fit on the training fold only;
    insert this transformer inside an sklearn Pipeline for cross-validation.
    Quality indicators and labels are deliberately separate from this API.
    """
    def fit(self, X: pd.DataFrame, y=None):
        self._validate(X)
        if len(X) == 0:
            raise ValueError('Cannot fit on an empty training set')
        self.feature_names_in_ = np.array(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        clean = X.replace([np.inf, -np.inf], np.nan).astype(float)
        self.dropped_ = {}
        retained = []
        for column in clean:
            values = clean[column]
            if values.notna().sum() == 0:
                self.dropped_[column] = 'all_missing'
            elif values.nunique(dropna=True) <= 1:
                self.dropped_[column] = 'constant'
            elif any(np.array_equal(values.to_numpy(), clean[other].to_numpy(), equal_nan=True) for other in retained):
                self.dropped_[column] = 'duplicate'
            else:
                retained.append(column)
        if not retained:
            raise ValueError('No nonconstant training features remain')
        self.columns_ = retained
        self.medians_ = clean[retained].median()
        filled = clean[retained].fillna(self.medians_)
        self.means_ = filled.mean()
        self.scales_ = filled.std(ddof=0).replace(0, 1)
        return self

    @staticmethod
    def _validate(X):
        if not isinstance(X, pd.DataFrame) or X.columns.duplicated().any():
            raise ValueError('Expected DataFrame with unique descriptor columns')
        if any(not (str(c).startswith('audio__') or str(c).startswith('facial__')) for c in X):
            raise ValueError('Only namespaced acoustic/facial descriptors are accepted')
        if any(any(token in str(c).lower() for token in ('phq', 'gender', 'participant', 'ptsd', 'pcl')) for c in X):
            raise ValueError('Forbidden predictor column')

    def transform(self, X):
        check_is_fitted(self, 'columns_')
        self._validate(X)
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError('Feature schema/order differs from fitted schema')
        clean = X[self.columns_].replace([np.inf, -np.inf], np.nan).astype(float)
        return (clean.fillna(self.medians_) - self.means_) / self.scales_

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'columns_')
        return np.array(self.columns_, dtype=object)


    @classmethod
    def from_state(cls, state: dict):
        """Restore the numeric JSON state emitted by the table builder; no pickle."""
        instance = cls()
        instance.feature_names_in_ = np.array(state['input_columns'], dtype=object)
        instance.n_features_in_ = len(instance.feature_names_in_)
        instance.columns_ = list(state['output_columns'])
        instance.dropped_ = dict(state['dropped'])
        cls._validate(pd.DataFrame(columns=instance.feature_names_in_))
        if not instance.columns_ or len(set(instance.columns_)) != len(instance.columns_) or not set(instance.columns_).issubset(instance.feature_names_in_):
            raise ValueError('Invalid saved feature selection')
        for name, key in [('medians_', 'medians'), ('means_', 'means'), ('scales_', 'scales')]:
            values = pd.Series(state[key], dtype=float).reindex(instance.columns_)
            if not np.isfinite(values).all():
                raise ValueError('Invalid saved numeric preprocessing state')
            setattr(instance, name, values)
        if (instance.scales_ <= 0).any():
            raise ValueError('Saved scales must be positive')
        return instance
