# Stage 5 protocol: learned adaptive fusion without quality inputs

Fixed on 2026-10-28 before real-data fitting. Target policy and participant
partitions are unchanged from Stage 4. Results remain provisional against the
supplied PHQ_Binary field. The official test is not predicted or scored.

## Architecture

- Independent audio and facial linear encoders into 16-dimensional ReLU embeddings.
- Encoder dropout 0.2 during training only.
- A gate taking concatenated embeddings: linear 32-to-8, ReLU, linear 8-to-2,
  softmax. The weights sum to one for each participant.
- Weighted sum of the two embeddings followed by a linear binary prediction head.
- No explicit quality indicators, metadata, labels, scores, or demographics enter
  the prediction path. Labels only supervise the training loss.
- Gate weights are mixing coefficients, not causal explanations or calibrated
  reliability estimates. The model does not implement Stage 6 quality-aware fusion.

## Selection and reproducibility

Five-fold stratified CV over official train, same CV seeds as Stage 4: 42 (primary),
137, 2026. Preprocessing fits separately on each training fold and is reused only
across candidates within that same fold. Model initialization uses seed+fold+1.
Candidate settings, fixed before evaluation: hidden=16, dropout=0.2, AdamW learning
rate=0.001, epochs in [50,100], weight decay in [0.001,0.01]. Full-batch training is
appropriate for 163 participant rows. Weighted BCE uses negative/positive class
count from the fitting fold. No development-based early stopping or calibration.
The candidate with highest mean positive-class CV F1 is selected (first wins ties).
It is refitted on all official training participants for its fixed epoch count,
using the run seed. Threshold is fixed at 0.5 for every evaluation.

Report all three seeds without selecting the best. CV selection F1 is not an
unbiased performance estimate; seed SD is not a confidence interval. Development
scores support iteration only, and any subsequent choices must remain transparent.

Use deterministic PyTorch algorithms, fixed NumPy/Python/Torch seeds, deterministic
cuDNN, disabled TF32, and CUBLAS_WORKSPACE_CONFIG=:4096:8. Cross-device or software-
version changes may still change numerical results. Checkpoints contain state_dict
plus JSON-compatible preprocessing state and are loaded with weights_only=True.
Persist fit/selection/inference time, CUDA peak tensor allocation (not total GPU
or system RAM usage), training losses, CV membership, gate statistics, probabilities,
metrics, environment and input/source hashes. Verify checkpoint predictions.

## Dependency

PyTorch 2.7.1 is pinned for this experiment, with a CUDA 12.8 wheel for the RTX 4050.
Installation source and available builds:
https://pytorch.org/get-started/previous-versions/#v271
No pretrained weights or dataset downloads are required.
