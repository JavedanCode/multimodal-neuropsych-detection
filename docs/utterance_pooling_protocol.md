# Learned MPNet utterance pooling

Declared before fitting, 2026-11-03. Reuse screened, frozen MPNet utterance
embeddings, all 161 training participants and unchanged released labels. Preserve
all utterances (41-197 per participant), padding with masks. Fixed Gaussian
projection 768->64 with seed42 and scale 1/sqrt(64) reduces training cost without
using labels or fitting to held-out features. Validate cache hashes and means.

Compare a matched mean-pooling network and learned attention-pooling network.
Both: 64->16 linear + tanh per utterance, then pooling and a linear binary head.
Attention adds a 16->1 learned score and masked softmax over utterances. The
attention model has 17 additional parameters; pooling order is invariant, not a
temporal sequence model. Standardize projected valid utterances using only the
current fitting fold. All utterances of a person stay together. One loss per
participant, balanced positive BCE weight, full-batch AdamW lr=.003, decay=.01.
No dropout, no threshold tuning. Candidate epochs 40 and 100, selected by pooled
inner F1. Full-batch training uses all participants, not utterance pseudo-labels.

Same five outer folds x seeds42/137/2026, three inner folds seed+fold+1. Train
each inner model once to100 epochs and record40/100. For each architecture choose
epochs on inner F1; also choose architecture+epoch jointly on inner F1. First
ties. Fit final model for selected epochs on outer training only. Threshold0.
Seed model initialization with repeat seed+fold; never select initialization.

Replay the historical MPNet linear-SVM reference on identical outer folds.
Primary candidate is the inner-selected neural procedure. Report both fixed
architectures regardless of results, domain metrics, all precision/recall/F1/AUC.
Improvement screen versus reference: F1 +3pp, no BA/AUC decrease, recall loss
<=5pp and F1 gain in >=2 repeats. Precision screen additionally reports +5pp
precision, recall>=65% and loss<=5pp, F1 loss<=2pp, gain in >=2 repeats.
Do not promote from an outer-best epoch or architecture. This is exploratory
reused training CV; no new dev/test evaluation or deployment. No new downloads.

Save per-fold selected settings, inner scores, fitting IDs, scaler parameters,
network weights and hashes. Verify mask behavior, all cache contracts, restored
predictions, selections, metrics and exact historical reference replay. Keep
the current reference if no candidate clears its screen.
