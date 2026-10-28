# Stage 8 completion: 2026-10-28

Completed run: `experiments/runs/stage8_20261028T180217572773Z`.
Source checkpoints: `experiments/runs/stage7_20261028T142146453803Z`.
All 18 matched Stage 7 models and all 56 development participants were included.
The original checkpoints and their predictions were not modified.

## Findings

- Top-ranked descriptor replacement produced greater mean absolute logit change
  than random replacement in all 54 model/fraction pairs (10%,25%,50%). This is
  aggregate evidence for reference-specific sensitivity, not success on every
  participant or a statistical significance claim.
- Example: all-feature quality model, primary seed 42, replacing the top 25% gave
  mean absolute logit change 0.4036, random 0.1582, bottom 0.0161.
- All 1,008 explanation rows passed the fixed completeness diagnostic. Maximum
  absolute residual was 0.000594 logits. The largest model-level mean L1 difference
  between 256-step and 512-step attribution vectors was 0.002445.
- All-feature global ranking correlations across seeds were about 0.12-0.29;
  top-10 Jaccard overlap was 0.11-0.33. HHO overlap was 0.05-0.11, with negative
  union-ranking correlations because masks and importance changed across seeds.
- L1 top-10 overlap was 0.82-1.00, but each model has only 11 selected features;
  this is not evidence of stable ordering. Its ranking correlations ranged from
  -0.14 to 0.35. Matched quality/adaptive rankings were nearly identical.

## Verification and limitations

78 tests passed without warnings. Tests cover exact linear and quadratic IG,
signed contributions, completeness, zero reference, known influential features,
random repeatability, unavailable-modality attribution, ranking alignment and
provenance mismatch rejection. All 18 restored predictions matched Stage 7.
Repeating the full explanation calculation produced 93 byte-identical output
artifacts (excluding the run metadata whose source hashes changed after an index
tie-handling correction). Details are in the run's verification.json.

An initial test and attempted run failed because reversed NumPy index arrays had
negative strides unsupported by PyTorch. Copying indices fixed that failure.
Bottom-ranking ties now explicitly preserve original feature order. The failed
run remains an incomplete ignored artifact, not a completed result.

Feature attributions are conditional on fixed participant quality. Separate
quality-mean replacement output changes are sensitivity checks, not additive
quality explanations. The reference is the saved training mean, not an actual
person. Perturbing correlated measurements may create unrealistic inputs.
The fixed midpoint integration check is numerical, not proof of meaningful
explanations. Gate weights are not presented as explanations. No feature names
are translated into clinical symptoms, and there is no causal claim.

The test partition remains unevaluated. Label semantics remain provisional.
No model/seed/settings were selected using explanation outcomes. Stage 9
(ablations and final evaluation) and Stage 10 (final reproducibility audit)
require separate continuation.
