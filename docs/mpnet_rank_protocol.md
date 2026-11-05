# Focused MPNet compression study

Fixed 2026-11-02 after the encoder comparison and before this study's fitting.
Same 161 training people, provisional released labels and screened frozen MPNet
features. No new embeddings, dev/test scoring or relabeling. The MPNet PCA32 SVM
is the stronger research reference (58.1% mean F1 in the preceding matched CV).

Test whitened standardized PCA ranks 8/16/32/64 and linear SVM only. Search
C .01/.1/1/10 and weights none/balanced. Same 5 outer folds x seeds 42/137/2026;
3 inner folds (seed+fold+1), pooled inner F1, first tie. Preprocessing fits within
each inner fitting split. Threshold choices 0/-.5/.5; default means zero.
Reproduce the old reference by restricting rank32 to C .01/.1/1 and zero cutoff.

Report each fixed-rank default/tuned procedure, the inner-selected rank procedure
at default cutoff (primary), and rank-plus-threshold selection (secondary).
Fixed-rank outer results are descriptive diagnostics, not an outer selection
rule. Neither their largest score nor a favorable repeat automatically becomes
the reference. Include all metrics/SD, constants, and AI/WoZ descriptions.
Primary/secondary promotion screen versus old reference: >=3 mean F1 points,
no balanced-accuracy/AUC loss, recall loss <=5 points, gain in >=2 repeats.
Reused training data remain exploratory. Stop after this registered comparison;
do not expand ranks, thresholds or C based on outer outcomes. Preserve negatives.
