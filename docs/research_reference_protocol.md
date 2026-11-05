# Full-training research reference

Defined 2026-11-02 after completing the exploratory comparisons and before the
full-training refit. This is an artifact-building step, not an additional model
comparison or performance estimate. Preserve the previous checkpoint artifacts.

Use screened frozen MPNet means, fit-only standardized whitened PCA32 and linear
SVM, zero margin cutoff. Retain the smaller successful C grid .01/.1/1 and class
weights none/balanced. Select C/weight on three participant folds with shuffle
seed43, pooled F1 and first tie; estimator seed42. Then refit preprocessing and
classifier on all 161 training participants. No development/test inputs or scores.

Save the pipeline under ignored models/research_reference_<UTC timestamp>, with
ordered input schema, target limitations, encoder revision/hashes, training IDs,
selected parameters, selection audit, source/input hashes and environment.
Reload the pipeline and reproduce its scores to verify serialization; do not
report fitting-set or tuning scores as new generalization performance.

The reference is chosen because MPNet SVM passed the earlier exploratory screen
and the follow-up searches failed to improve it. This is not evidence of a global
optimum or independent test validation. No deployed application is changed.
