# Stage 6 protocol: quality-conditioned adaptive fusion

Fixed on 2026-10-28 before real-data fitting. Same released PHQ_Binary target,
participant summaries, official splits, training-only CV, seeds, threshold, loss,
optimizer and search grid as Stage 5. Results remain provisional. No test scoring.

## Mechanism

Reuse Stage 5's acoustic/facial encoders and prediction head. Add a zero-initialized
8-by-8 quality projection to the first gate layer's preactivation. Center each
quality field using its fitting fold's mean (no labels used). Quality inputs go
only to this gate correction; they do not enter the encoders or classifier head.
Zero initialization consumes no random draws, preserving Stage 5's initial
encoder/gate/head weights and dropout random stream. This adds 64 parameters.

Only these fixed, source-derived indicators are accepted, in a fixed order:
audio availability, invalid descriptor fraction, retained-row fraction; facial
availability, invalid descriptor fraction, retained-row fraction, mean extractor
confidence, valid-confidence fraction. Require finite values in [0,1]. Availability
is binary. Validate participant IDs and reject extra columns, including labels or
demographics. No target-derived quality indicator is constructed.

If one modality is unavailable, suppress its input/embedding and force its gate
weight to zero. Reject both unavailable. This is a defensive inference rule, not
evidence of learned robustness: all current participants have both modalities.

Stage 3 recorded only two varying quality indicators (facial frame retention and
mean confidence); the other six are constant. Report constants from training data
and center them to zero. The observed corpus cannot test improvements based on
varying audio quality, or naturally absent modalities. Confidence is not a
calibrated clinical reliability estimate. The learned quality correction need not
be monotonic with confidence and cannot be interpreted as clinical explanation.

## Evaluation

Use 5 stratified training folds and seeds 42 (primary), 137, 2026. Inside each fold,
fit descriptor preprocessing and quality means only on its fitting participants.
Use 16-dimensional embeddings, dropout 0.2, AdamW learning rate 0.001, candidate
epoch counts [50,100], weight decays [0.001,0.01], fold-derived BCE positive weight.
Select mean positive-class training-CV F1, first candidate on ties. Refit on
training only. Fixed threshold 0.5. No development-based early stopping, parameter
selection, seed selection, calibration, or quality-threshold tuning.

Verify checkpoint predictions and primary CUDA retraining. Compare input hashes,
CV participant membership, and search settings to Stage 5 before reporting paired
results. Report all seeds, not just the strongest. A constant-quality synthetic
control must reproduce Stage 5 training numerically. Full perturbation experiments,
matched ablation studies and final held-out evaluation remain later stages.

Changing the shared runner must preserve Stage 5 compatibility and all existing
tests. Quality inputs must remain optional and disabled for the Stage 5 CLI.
