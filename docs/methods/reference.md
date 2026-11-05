<!-- Original frozen protocol: 5f3e3f3c4f754c25e92f377cdfedcf822d561a71:docs/pipeline_v2_protocol.md -->
# Pipeline remediation v2

Declared 2026-11-03 before v2 experiments. Historical source and caches remain
frozen for reproducibility; new work uses explicit v2 entrypoints. Preserve the
161-person matched training cohort and provisional released labels. No new
official development/test evaluation and no production deployment.

Order: (1) explicit unknown-confidence masks; (2) joint label/domain evaluation
and WoZ-to-AI training-only transfer; (3) inner learning curves/early epoch
selection; (4) utterance-aligned MPNet/audio/facial features and measured quality;
(5) local speaker/timing review package and unresolved provenance record.

Evaluation: five outer folds x seeds42/137/2026 jointly stratified by label and
manual-defined interview domain; three inner folds with seed+fold+1. Each outer
evaluation fold contains one AI positive. Fit preprocessors and select settings
only on inner training data. Keep original cohort/folds results separately;
new-fold numbers are not improvements over old-fold numbers. Transfer diagnostic
fits WoZ only and evaluates the 18 training-partition AI interviews; it is not
a new independent test cohort. IDs/domain never enter predictors.

Reference: MPNet mean -> fold-fitted standardized whitened PCA32 -> linear SVM,
C=.01/.1/1 and weights None/balanced, pooled inner F1, first ties, cutoff0.
For neural learning curves: fixed projected MPNet utterances as in the previous
pilot; matched mean/attention with shared encoder/head initialization, dropout
.2, AdamW lr .003, weight decay .01 or .1, balanced participant BCE. Record
epochs5/10/20/40/100, choose architecture/decay/epoch using pooled inner F1.
Save fitting/inner-validation loss and F1 curves. Outer labels cannot select an
epoch. Report every architecture and the inner-selected procedure.

Alignment: retain every screened utterance boundary and its original MPNet
embedding. Aggregate actual eGeMAPS/OpenFace timestamps inside [start,end).
Keep endpoint-mismatched facial input unavailable; no invented time offsets.
Measured audio quality includes clipping, RMS, short-frame energy variation,
zero-valued sample fraction and coverage. These are observable proxies, not
validated SNR/diarization or calibrated reliability probabilities. Unknown ASR
confidence has a separate known flag. Do not claim speaker validation from
transcript intervals. Do not hand corruption severity to a quality gate.

For new candidates retain F1 screen (+3pp, no BA/AUC loss, recall loss<=5pp,
F1 gain in>=2 repeats). Also report precision/recall, average precision and
domain outcomes. No post-hoc winner from a changed target, split or seed.

Manual inspected locally: data/raw/E-DAIC Manual.pdf, SHA256
eff6958ac918492b20cae9506e3f69e2a87f0fd5ce9cf1382ca5da16b634e706.
It confirms mixed WoZ/AI train/dev and AI-only test, and the ID domain ranges.
It lists PHQ_Binary/PHQ_Score but does not define their reconciliation or promise
participant-only transcripts. It describes219 directories versus this release's275.
The label and speaker questions therefore remain unresolved, not fixed by
guessing. Existing provider inquiry stays unsent without explicit send authority.
