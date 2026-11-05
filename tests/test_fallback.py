import numpy as np
import pytest
from src.models.fallback import design, from_scores


def data():
    q = np.ones((2, 11))
    q[:, 2] = 0
    q[:, 5] = 0
    return {"quality": q, "available": np.ones((2, 3))}


class Meta:
    def decision_function(self, x):
        return x[:, 0] + x[:, 1]


def test_primary_design_never_uses_text_scores_or_text_quality():
    d = data()
    s = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    for quality in [False, True]:
        before = design(s, d, quality)
        s[:, 0] = 1e8
        d["quality"][:, 9:] = 1e9
        np.testing.assert_array_equal(design(s, d, quality), before)


def test_fallback_is_only_for_absent_or_ineligible_auxiliaries():
    d = data()
    s = np.array([[9.0, 2.0, 3.0], [8.0, 5.0, 6.0]])
    values, route = from_scores(Meta(), s, d, True, True)
    assert not route.any()
    np.testing.assert_array_equal(values, [5.0, 11.0])
    values, route = from_scores(Meta(), s, d, True, True, "missing_both")
    assert route.all()
    np.testing.assert_array_equal(values, [9.0, 8.0])
    d["quality"][0, 2] = 0.5
    d["quality"][0, 6] = 0.1
    values, route = from_scores(Meta(), s, d, True, True)
    np.testing.assert_array_equal(route, [True, False])
    assert values[0] == 9.0
    d["available"][0, 0] = 0
    with pytest.raises(ValueError):
        from_scores(Meta(), s, d, True, True)


def test_inference_does_not_read_text_when_primary_input_is_usable():
    from src.models.fallback import predict

    class Base:
        def decision_function(self, x):
            return np.ones(len(x))

    class ForbiddenText:
        def decision_function(self, x):
            raise AssertionError("Text must not be evaluated")

    d = data()
    d.update(audio=np.zeros((2, 1)), face=np.zeros((2, 1)), text=None)
    bundle = {
        "quality": True,
        "fallback": True,
        "meta": Meta(),
        "base": {"audio": Base(), "face": Base(), "text": ForbiddenText()},
    }
    values, routes = predict(bundle, d)
    assert not routes.any()
    np.testing.assert_array_equal(values, [2.0, 2.0])
