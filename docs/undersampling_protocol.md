# MPNet undersampling experiment

Declared before fitting, 2026-11-03. Keep the existing 161 participants, released
labels, five outer folds and seeds 42/137/2026. Never sample evaluation rows.
Use the same three inner folds and MPNet PCA32 preprocessing. Fit preprocessing
on all fitting-fold participants before sampling classifier rows; this deliberately
isolates classifier resampling and retains training-only covariance information.
No development/test predictions or new encoder extraction.

Reference: original SVM C=.01/.1/1 and weights None/balanced, inner pooled F1.
Additional control: balanced weights only. Candidates: single random negative
subset at 1:1 and 2:1 negatives:positives; seven-member averaged-margin ensembles
at each ratio. All positives retained, sampling without replacement per member;
members can overlap. Allocate negative quotas proportionally by AI/WoZ domain
(existing participant-ID domain convention), using largest remainders. Domain is
used only for sampling, never as a feature. Deterministic seeds fixed independently
of outcomes. C=.01/.1/1 selected separately per procedure on pooled inner F1.
Sampled classifiers use no class weighting. Zero margin threshold throughout.
Primary candidate chooses among the four sampling procedures using inner F1;
report every procedure, including negative results. First candidate wins ties.

Screen against original reference: at least +3 percentage points mean F1,
no mean balanced-accuracy or AUC decrease, recall loss no greater than 5 points,
and F1 improvement in at least two of three repeats. Exploratory reused-fold
comparison; no independent superiority claim. Save sample IDs, inner scores,
selected settings, source/input hashes and all fitted bundles. Verify exclusion
of held-out IDs, quotas, positive retention, restored predictions, metrics and
exact reproduction of the previous MPNet reference.
