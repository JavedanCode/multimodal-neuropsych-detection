# Complementary text-encoder ensemble

Declared before fitting, 2026-11-03, after the learned-pooling negative result.
Use the unchanged screened MPNet and MiniLM participant means, same 161 people,
five outer folds x seeds42/137/2026 and three inner folds seed+fold+1.
Each encoder gets its own fitting-only standardization and whitened PCA32.
Each classifier is linear SVM with C=.01/.1/1 and weights None/balanced.

Choose both classifiers and MPNet blend weight jointly by pooled inner F1.
Weights tested in order: 1, .75, .5, .25, 0. Score is w*MPNet margin +
(1-w)*MiniLM margin, threshold0. These are raw SVM margins, not calibrated
probabilities. Endpoint candidates use just the relevant encoder. Ties keep
first candidate, preferring MPNet-only. Report original MPNet reference and
the inner-selected blend; no outer selection of combination or hyperparameters.
Sampling, thresholds, labels, participants and encoder weights are unchanged.

Screen: mean F1 +3pp, no BA/AUC decrease, recall loss<=5pp and positive F1
differences in at least two repeats. Report precision, recall, F1, AUC, domain
metrics and all selected weights, including when selection returns a single
encoder. Exploratory reused-fold estimate, not new dev/test evaluation.
Save inner per-encoder scores, all candidate F1 values, selection audits,
pipelines, predictions and hashes. Independently recompute every combination,
verify selection, restore predictions and exactly reproduce the old control.
