"""Guard the fold schemes and the confidence gate of the stacked fusion."""

import numpy as np
import pandas as pd
import pytest

from src.training.stacked_study import outer_splits


def cohort():
    ids = list(range(300, 420)) + list(range(600, 640))
    random = np.random.default_rng(0)
    labels = pd.Series((random.random(len(ids)) < 0.3).astype(int), index=ids)
    gender = pd.Series(np.where(random.random(len(ids)) < 0.45, "female", "male"), index=ids)
    return labels, gender


@pytest.mark.parametrize("scheme,count", [("A", 5), ("B", 5), ("C", 10)])
def test_each_person_is_held_out_exactly_once(scheme, count):
    labels, gender = cohort()
    folds = outer_splits(labels, gender, scheme, 42)
    assert len(folds) == count
    held = np.concatenate([b for _, b in folds])
    assert sorted(held) == list(range(len(labels)))
    for fitting, heldout in folds:
        assert not set(fitting) & set(heldout)
        assert labels.iloc[heldout].sum() > 0


def test_gender_scheme_balances_gender_at_least_as_well():
    labels, gender = cohort()

    def spread(scheme):
        return np.std(
            [
                (gender.iloc[b] == "female").mean()
                for seed in (42, 137, 2026)
                for _, b in outer_splits(labels, gender, scheme, seed)
            ]
        )

    assert spread("B") <= spread("A")


def test_unknown_scheme_is_rejected():
    labels, gender = cohort()
    with pytest.raises(ValueError):
        outer_splits(labels, gender, "D", 42)


def test_cli_stacked_study_runs_alone():
    from src.__main__ import main

    with pytest.raises(SystemExit):
        main(["run", "audio-visual", "--stacked-study", "--pooled-study"])
