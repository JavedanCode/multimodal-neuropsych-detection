"""Apply predeclared paired improvement screens without selecting test winners."""

MEASURES = [
    "f1",
    "precision",
    "recall",
    "specificity",
    "accuracy",
    "balanced_accuracy",
    "mean_fold_auc",
]


def paired(metrics, candidate, control, scenario="clean"):
    sub = metrics[metrics.scenario == scenario].set_index(["condition", "seed"])
    a, b = sub.loc[candidate].sort_index(), sub.loc[control].sort_index()
    assert a.index.equals(b.index)
    differences = a[MEASURES] - b[MEASURES]
    means = differences.mean()
    passes = (
        means.f1 >= 0.03
        and means.balanced_accuracy >= 0
        and means.mean_fold_auc >= 0
        and means.recall >= -0.05
        and (differences.f1 > 0).sum() >= 2
    )
    return {
        "candidate": candidate,
        "control": control,
        "scenario": scenario,
        "mean_differences": means.to_dict(),
        "f1_positive_repeats": int((differences.f1 > 0).sum()),
        "passes_clean_screen": bool(passes),
        "repeat_differences": differences.reset_index().to_dict("records"),
    }
