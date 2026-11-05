"""Reusable helpers extracted from the recorded research experiments."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from src.dataset.preprocessing import ParticipantPreprocessor


class TextPreprocessor(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.input_columns_ = list(X)
        self.text_ = [c for c in X if c.startswith("text__")]
        self.other_ = [c for c in X if c not in self.text_]
        self.columns_ = []
        if self.text_:
            if not np.isfinite(X[self.text_].to_numpy()).all():
                raise ValueError("Nonfinite text")
            self.scale_ = StandardScaler().fit(X[self.text_])
            self.pca_ = PCA(
                n_components=min(32, len(X) - 1, len(self.text_)),
                svd_solver="randomized",
                random_state=42,
                whiten=True,
            ).fit(self.scale_.transform(X[self.text_]))
            self.columns_ += [f"text__pc{i}" for i in range(self.pca_.n_components_)]
        if self.other_:
            self.base_ = ParticipantPreprocessor().fit(X[self.other_])
            self.columns_ += self.base_.columns_
        return self

    def transform(self, X):
        check_is_fitted(self, "columns_")
        if list(X) != self.input_columns_:
            raise ValueError("Schema mismatch")
        arrays = []
        if self.text_:
            arrays.append(self.pca_.transform(self.scale_.transform(X[self.text_])))
        if self.other_:
            arrays.append(self.base_.transform(X[self.other_]).to_numpy())
        values = np.concatenate(arrays, axis=1)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite projection")
        return pd.DataFrame(values, index=X.index, columns=self.columns_)
