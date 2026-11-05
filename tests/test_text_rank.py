import numpy as np
import pandas as pd
from src.models.text_preprocessing import TextPreprocessor
from src.models.text_rank import TextRankPreprocessor


def test_rank32_reproduces_original_preprocessing():
    rng=np.random.default_rng(10)
    x=pd.DataFrame(rng.normal(size=(60,80)),columns=[f'text__{i}' for i in range(80)])
    original=TextPreprocessor().fit(x.iloc[:50])
    new=TextRankPreprocessor(32).fit(x.iloc[:50])
    np.testing.assert_array_equal(original.transform(x.iloc[50:]),new.transform(x.iloc[50:]))


def test_rank_control_and_heldout_independence():
    rng=np.random.default_rng(11)
    x=pd.DataFrame(rng.normal(size=(60,80)),columns=[f'text__{i}' for i in range(80)])
    prep=TextRankPreprocessor(8).fit(x.iloc[:50])
    expected=prep.transform(x.iloc[:50])
    x.iloc[50:]=1e6
    assert expected.shape==(50,8)
    np.testing.assert_array_equal(expected,prep.transform(x.iloc[:50]))
