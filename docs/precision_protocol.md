# Precision experiment series

Frozen before fitting, 2026-11-03. Execute four experiments sequentially, using
the unchanged 161-person MPNet cohort and official provisional labels. Preserve
five outer folds, seeds 42/137/2026 and three inner folds (seed+fold+1). All
preprocessing is fit inside the fitting fold. No development/test predictions,
label changes, encoder changes, or production deployment.

Reference in every outer fold: existing PCA32 + linear SVM, C=.01/.1/1,
weights None/balanced, maximize pooled inner F1, first ties, threshold zero.

1. Threshold: keep reference parameters, choose margin cutoff among
   0, .1, .2, .3, .5, .75, 1 using inner predictions only.
2. Cost-sensitive SVM: linear SVM C=.01/.1/1; weights {0:1,1:w} for
   w=.75/1/1.5/2/3; zero cutoff. All training participants retained.
3. Nonlinear SVM: RBF C=.1/1/10; gamma=scale/.01/.1;
   weights None/balanced; zero cutoff, unchanged PCA32.
4. Agreement: retain reference SVM and require a logistic regression also to
   predict positive. Logistic C=.01/.1/1, weights None/balanced. Use minimum
   of the two raw margins, zero cutoff. Raw margins are not probabilities;
   AUC for this condition describes this declared minimum-margin ranking.

For each experiment maximize pooled inner precision subject to recall >=.65
and F1 >= reference inner F1-.02. Break ties by F1, then recall, then candidate
order. If none qualifies, use the reference unchanged. Keep a reference
candidate available when it qualifies. This is a constrained precision policy,
not a promise that outer recall will meet the inner constraint.

Report every experiment, false positives/negatives, AI/WoZ summaries and repeat
variation. Exploratory outer improvement screen: precision gain >=5 percentage
points, recall loss <=5 points and mean recall >=.65, F1 loss <=2 points,
and precision gain in at least two repeats. No post-hoc threshold selection.
The screen identifies a precision tradeoff candidate, not independent proof or
an automatic replacement of the best-F1 checkpoint. Preserve all results.

Save inner scores, selected candidates, folds, models and source/input hashes.
Independently check selections, restore all predictions, recompute metrics,
check preprocessing membership and reproduce the original reference scores.
