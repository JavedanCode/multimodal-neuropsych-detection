"""Reusable helpers extracted from the recorded research experiments."""

from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from src.models.text_preprocessing import TextPreprocessor


class LexicalPreprocessor:
    def __init__(self, semantic_weight=0.0):
        self.semantic_weight = semantic_weight

    def fit(self, data):
        self.tfidf = TfidfTransformer(sublinear_tf=True).fit(data["counts"])
        if self.semantic_weight:
            self.semantic = TextPreprocessor().fit(data["semantic"])
        return self

    def transform(self, data):
        lexical = self.tfidf.transform(data["counts"])
        if not self.semantic_weight:
            return lexical
        semantic = (
            normalize(self.semantic.transform(data["semantic"]).to_numpy())
            * self.semantic_weight
        )
        return sparse.hstack([lexical, sparse.csr_matrix(semantic)], format="csr")
