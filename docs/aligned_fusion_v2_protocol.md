# Aligned measured-quality fusion v2

Declared before fitting. Use completed aligned_v2 training features and the same
joint label/domain folds as baseline_v2. All 161 participants and original labels
remain. Participant MPNet mean is the original verified mean. Audio and face
features are means and standard deviations across available utterance summaries.
Quality features summarize the 11 observed utterance-quality fields. Availability
is explicit. No IDs, domain, PHQ scores, or questionnaire answers are predictors.

Three base predictors: PCA32 + balanced linear SVM C=.1 for text; standardized
balanced logistic C=.1 for audio and face. Construct base out-of-fold margins
inside each fitting set using three joint-stratified folds when counts allow,
otherwise two joint-stratified folds (never a class-only silent fallback).
Fit base models on the full fitting set for subsequent inference. Meta learners
see only out-of-fold base margins, not fitted-participant margins. This internal
cross-fitting is repeated inside each inner fold of outer model selection.

Compare: text-margin calibration control, ordinary stacking, and measured-quality
stacking. Meta logistic C=.01/.1/1, balanced weights, cutoff0. Ordinary input is
three margins plus audio/face availability. Quality adds per-modality margin x
measured-quality interactions; quality is not a diagnostic feature on its own.
Training augmentation includes missing audio, missing face, and both missing;
each of four copies has weight.25. Missing scores/quality/availability are zeroed
together. Quality proxies are not calibrated reliability probabilities, and this
is conditional stacking rather than a claim of a novel neural gating mechanism.

Choose meta C per procedure by pooled inner F1. The primary selected procedure
chooses among all three plus the original fold-selected MPNet baseline using
inner F1, with baseline first on ties. Report all fixed procedures regardless.
Evaluate clean and the three missing-input scenarios with observable availability;
no injected corruption-severity hints. Do not claim robustness to real audio noise
or speaker correctness from these tests. No new official dev/test predictions.

Primary clean screen: F1 +3pp against the matched baseline, no BA/AUC loss,
recall loss<=5pp, improvement in>=2 repeats. Quality-vs-ordinary robustness screen:
clean F1 loss<=2pp and mean missing-input F1 gain>=5pp. Report all scenarios and
available text-only control, mean fold AP, nested selections, domains and hashes.
