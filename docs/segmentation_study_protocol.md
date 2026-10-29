# Transcript coverage and facial fusion pilot

Frozen 2026-10-29 before feature extraction/model fitting. Exploratory training-only
study; Stage 10 remains pending. Labels remain unchanged released PHQ_Binary.
No development/test predictions or changes to previous experiments.

Exclude training recordings whose previously audited WAV/feature endpoints differ
by more than one second, identically in every condition. Record IDs only in ignored
artifacts. Verify audit archive stats and source hashes before using its duration.
This quarantine does not resolve the underlying source mismatch.

Compare full versus transcript-covered audio mean/population-standard-deviation
features (46), each alone and concatenated with the existing 98 full-session facial
summaries. Also include facial-only and constant-zero/constant-one references.
Facial tracking filtering remains unchanged. These are session-level combinations,
not synchronized facial/audio sequences. Timestamp intervals are merged, half-open,
nonempty, and bounded by the audited WAV duration. Use actual frameTime, preserving
duplicates. Keep zero values; ignore nonfinite descriptor values. No speech identity
or detailed timing validation has been performed: this tests transcript coverage,
not verified participant-only speech. Do not infer silence from missing transcripts.

Five matched stratified outer folds, seeds 42/137/2026; three inner folds seeded
outer seed + fold index + 1. Two classifiers: logistic regression and linear SVM,
C=[0.01,0.1,1], class_weight=[None,balanced]. Select by pooled inner positive F1,
first tied candidate wins. Fixed zero decision margin cutoff; no threshold search.
All preprocessing fits within fitting folds. Save inner candidate scores/IDs,
pipelines, predictions, provenance and restored-model verification. No ID, domain,
scores or quality metadata as predictors.

Report every condition: pooled outer F1, precision, recall, specificity, balanced
accuracy, accuracy, and mean outer-fold AUC (never pooled SVM margins). Repeat SD
is not a confidence interval. Paired coverage/fusion differences use identical
participants/folds. Report WoZ/AI subgroup confusion metrics with sample counts;
tiny AI subgroup estimates are descriptive. No post-result additional searches,
new test evaluation, deployment, or guaranteed improvement. Any apparent gain
requires independent confirmation; prior results informed this pilot.
