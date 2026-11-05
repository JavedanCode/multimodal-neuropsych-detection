# Stage 8: descriptor attribution and perturbation checks

Fixed before examining explanation results on 2026-10-28. Explain all 18 Stage 7
checkpoints (all/L1/HHO, adaptive/quality, seeds 42/137/2026) on all 56 development
participants. No fitting, model selection, threshold changes or test evaluation.
Require Stage 7 input hashes and training participant IDs to match current data.

Use Integrated Gradients (IG), Sundararajan et al. (2017),
https://proceedings.mlr.press/v70/sundararajan17a.html . This is a gradient-based
alternative to SHAP for these differentiable networks, not a SHAP implementation.
Explain the positive-class logit, not a calibrated clinical risk probability.
The reference is zero in the checkpoint's standardized descriptor space: the
training mean after imputation. Keep each participant's quality inputs fixed.
Thus descriptor attributions explain f(x,q)-f(reference,q), not quality's effect.
Use midpoint integration at 256 and 512 steps on CPU, evaluation mode, one thread.
Report both discretization agreement and completeness residuals. Flag rows where
absolute residual exceeds 0.001 + 0.01*abs(output-reference output); this is a
numerical diagnostic, not an acceptance threshold for scientific usefulness.
Do not change steps or explanations based on development results.

Save signed per-participant attributions, actual/reference logits and probabilities,
global mean absolute importance, audio/facial sums and top measurements with exact
feature names. Positive attribution raises the positive-class logit relative to
the reference. Global mean absolute importance hides direction; retain both.
Do not equate gate weights with attribution or infer symptoms from feature names.

Faithfulness: replace ceil(10%,25%,50%) of retained descriptors by the same reference.
Rank separately for each participant by absolute IG, with stable column-order ties.
Compare top-ranked, bottom-ranked and 32 seeded random subsets of identical size.
Report absolute logit change (primary), absolute probability change and prediction
flip rate; average random trials within participant before aggregating people.
Keep quality fixed and recompute the gate on every perturbed input. This is a
reference-replacement sensitivity test, not a causal intervention; correlated
features, cancellation, interactions and unrealistic combinations can distort it.

For quality models, separately replace continuous quality fields by saved training
means, keeping binary availability fixed; report output changes, not additive IG
quality explanations. This does not test real missing-modality robustness.

Across seeds within each selector/model, compare global rankings on the union of
retained descriptors (absent descriptors assigned zero), Spearman correlation and
top-10 Jaccard overlap. This combines selection and attribution variability. Save
numerical 256/512 agreement separately. Seed variation is not a confidence interval.
Compare adaptive and quality models within selector/seed using the same summaries.

All results are exploratory, conditional on reference choice, whole-interview
aggregation and provisional released PHQ_Binary semantics. No clinical, causal,
benchmark superiority or explanation-faithfulness guarantee follows from these tests.
