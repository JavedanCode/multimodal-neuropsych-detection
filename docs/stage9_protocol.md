# Stage 9: frozen provisional held-out evaluation

Frozen on 2026-10-28 before test prediction. The user explicitly authorized final
evaluation against unchanged released PHQ_Binary despite unresolved discrepancies.
This evaluates reproduction of that field, not confirmed PHQ-8 >=10 screening.
Do not derive replacement labels, compare published headline scores, or claim
clinical validity. The label issue persists; authorization does not resolve it.

## Fixed model roster and training

- All nine Stage 4 logistic checkpoints: audio, facial, concatenation x seeds
  42/137/2026. Their original training-only CV choices remain fixed.
- All 18 Stage 7 neural checkpoints: all/L1/HHO descriptors x adaptive/quality
  fusion x the same three seeds. No refitting or selection using development/test.
- Nine new equal-weight neural controls: all/L1/HHO x three seeds, reusing Stage 7
  selected columns and exactly its fixed 50-epoch neural settings. Both modality
  embeddings receive weight 0.5. Remove the gate parameters; preserve encoder/head
  initialization and dropout RNG ordering against the adaptive architecture.
  Train only on the original 163 training participants. No hyperparameter search.
- One training-majority reference. Total: 37 evaluated models including majority.
- No training+development refit, ensembling, probability calibration, threshold
  tuning, new seed, or model choice based on final scores. Primary seed is 42.

Quality ablation: quality versus adaptive at matched selector/seed. Gate ablation:
adaptive versus equal-weight at matched selector/seed. Feature-selection ablation:
HHO versus all (and conventional L1 versus all) at matched architecture/seed.
Removing the adaptive gate also removes the quality mechanism; compare equal to
adaptive for the gate-specific effect, not equal to quality as a single-factor
ablation. Classical concatenation is a contextual baseline, not an isolated gate
ablation. Optional attention is not introduced.

## Freeze boundary and provenance

The prepare command verifies data hashes and source checkpoints, trains only the
nine equal controls, checks development prediction restoration, and writes a
manifest containing all checkpoint paths/hashes and source/protocol/data hashes.
Only after tests and preparation succeed may the evaluate command load test rows.
It verifies the frozen manifest and refuses to overwrite a completed evaluation.
All selected descriptors and preprocessing remain training-fitted. Audit official
participant partitions and schema before use. Never write raw data or publish
participant outputs. All run artifacts remain ignored locally.

## Metrics, uncertainty and comparisons

Evaluate both development and test, reporting them separately. Positive class 1,
threshold 0.5, precision, recall, F1, macro-F1, ROC-AUC, accuracy, specificity and
confusion matrices ordered [0,1]. Report every seed, plus mean/population SD across
seeds. SD is optimization variability on one cohort, not a confidence interval.

Primary-seed F1 intervals: 2,000 stratified participant bootstrap resamples,
random seed 91042, sample each class with replacement retaining original counts,
percentile 2.5/97.5. Conditional on observed class prevalence and trained model;
no retraining uncertainty, no multiple-comparison correction, no significance
or superiority claims. Use the same indices for paired primary-seed F1 difference
intervals for the predeclared quality, gate and feature-selection comparisons.
All-seed ablation deltas are also reported; never pick the best seed.

Report retained feature counts, checkpoint parameter counts for neural models,
original fitting/selection times when recorded and new inference timings; do not
equate incomparable historical timings to controlled speed benchmarks. Include
test F1 comparison and primary confusion-matrix figures, SVG and PNG, with fixed
0-1 F1 scale and explicit provisional target labels. Include all models in tables.
Test data are used only for prediction, metrics and uncertainty, never decisions.

## Interpretation

Whole-interview summaries, no diarization/temporal modeling, small participant
counts, class imbalance, seed sensitivity and unresolved target semantics limit
generalization. Stage 8 explanations remain exploratory development-only evidence.
Weak or negative results must be preserved. Final evaluation does not imply that
the research hypothesis succeeds or that the system is ready for clinical use.
