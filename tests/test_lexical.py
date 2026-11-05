import numpy as np
import pandas as pd
from scipy import sparse
from src.dataset.lexical import counts
from src.models.lexical import LexicalPreprocessor


def test_hashing_does_not_cross_utterance_boundaries():
    assert counts(["alpha beta", "gamma delta"]).sum() == 6
    assert counts(["alpha beta gamma delta"]).sum() == 7
    assert (counts(["alpha beta"]) != counts(["ALPHA BETA"])).nnz == 0


def test_idf_uses_only_fitting_participants():
    fit = {"counts": sparse.csr_matrix([[1, 0, 0], [1, 1, 0]]), "semantic": None}
    prep = LexicalPreprocessor().fit(fit)
    expected = np.log((1 + 2) / (1 + np.array([2, 1, 0]))) + 1
    np.testing.assert_allclose(prep.tfidf.idf_, expected)
    before = prep.tfidf.idf_.copy()
    prep.transform({"counts": sparse.csr_matrix([[0, 0, 100]]), "semantic": None})
    np.testing.assert_array_equal(prep.tfidf.idf_, before)


def test_semantic_relative_weight_preserves_sparse_lexical_branch():
    rng = np.random.default_rng(42)
    data = {
        "counts": sparse.csr_matrix(rng.integers(1, 5, size=(8, 5))),
        "semantic": pd.DataFrame(
            rng.normal(size=(8, 4)), columns=[f"text__{i}" for i in range(4)]
        ),
    }
    prep = LexicalPreprocessor(2).fit(data)
    transformed = prep.transform(data)
    assert sparse.issparse(transformed) and transformed.shape == (8, 9)
    np.testing.assert_allclose(
        np.asarray(transformed[:, :5].power(2).sum(1)).ravel(), 1
    )
    np.testing.assert_allclose(
        np.asarray(transformed[:, 5:].power(2).sum(1)).ravel(), 4
    )
