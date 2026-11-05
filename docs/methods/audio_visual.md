<!-- Original frozen protocol: 5f3e3f3c4f754c25e92f377cdfedcf822d561a71:docs/branch_audit_v6_protocol.md -->
# Primary branch audit and matched fusion — 2026-11-04

Freeze before fitting. Same161 retained training participants and released labels;
no official dev/test evaluation, new data, label changes, or text predictors.
Use joint domain/label outer5/inner3 folds, seeds42/137/2026. Do not choose a
representation or fusion recipe from outer results. Previous studies remain frozen.

Reuse existing feature caches: original whole-interview audio46/face98; v2 aligned
utterance audio92/face196; rich whole-interview audio230/face490. Rich summaries
already include quantiles and temporal variation. This comparison isolates them on
the revised cohort/folds and does not repeat the prior extraction. For consistency
apply the existing two facial alignment quarantines to all facial representations;
set those rows missing. Report this restriction rather than claiming exact replay
of older whole-interview experiments. Labels/IDs/quality are not branch predictors.

Shared fold-fitted preprocessing: reject forbidden schema, drop train-empty columns,
median impute, drop constant/exact duplicate imputed columns, standardize. No PCA.
For each of six representations choose logistic or linear SVM, C .01/.1/1,
balanced weights, cutoff0, pooled inner F1, first tie. Report each representation.
Within each outer training fold select one audio and one face representation from
their inner scores (original first on ties). Report these two selected branch
policies separately; they are not the post-hoc best outer-scoring representations.

Fusion uses those same two inner-selected representations, preprocessed on outer
training only. Compare: original-feature concatenation versus selected-feature
concatenation, each fixed balanced logistic C=.1; matched neural equal-weight,
adaptive gate, and quality-conditioned gate. Neural recipe fixed before fitting:
two16-unit ReLU encoders, dropout.2,8-unit gate, linear head, weighted BCE,
AdamW lr.003/weight decay.1,40 full-batch epochs, CPU deterministic seeds.
Quality input is the first9 observed v2 quality fields only, standardized using
outer-training statistics, zero-initialized extra gate weights. Explicit modality
availability masks are shared by all neural controls. Match encoder/head/gate
initialization and dropout RNG across neural ablations. No learned stacking or
OOF margins train the neural head: it learns directly from fitting-participant
features, then evaluates held-out people. Keep training curves as diagnostics.

13 fitted conditions: six representations, selected_audio, selected_face,
original_concat, selected_concat, neural_equal, neural_adaptive, neural_quality.
Also report always-negative and always-positive constants with scores -1/+1.
All configurations/choices stay within outer training. No new threshold search,
seed selection, encoder fine-tuning or text fallback in this representation audit.
Fallback functionality was already tested in v5; establish primary signal first.

Primary comparisons: aligned/rich against original within each modality; selected
branches against original; selected_concat against original_concat; adaptive versus
equal; quality versus adaptive; neural conditions versus original_concat. Original
F1 screen: +3pp, no balanced-accuracy/AUC reduction, recall loss<=5pp, F1 gain in
>=2 repeats. Report precision/recall/F1/AUC/AP, domains, confusion counts, feature
counts, timings and all candidate decisions. Always-positive F1 is37.4% here;
beating a weak model does not alone establish a useful classifier. These are
exploratory reused-cohort estimates, not fresh independent validation.

Stop after this finite comparison and verification. Preserve the best supported
previous reference if candidates fail. Document evidence for representation loss
or weak signal only to the extent supported; no proof of a dataset performance ceiling.
