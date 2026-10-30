# Text, temporal, and reliability follow-up

Fixed 2026-10-30 before fitting. Same 161 official training participants and two
timing quarantines as the previous pilots; unchanged provisional released binary
labels. No development/test evaluation or label substitution. Stage 10 pending.

Use local frozen sentence-transformers/all-MiniLM-L6-v2, pinned downloaded revision
and file hashes. Attention-mask mean pooling and L2 normalization, 256 tokens per
chunk; split long utterances into consecutive 254-token chunks without dropping
the remainder. Average chunks within utterance. No remote inference or code.
Source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

Transcript audit: require timestamps and Text, report available speaker columns,
missing text/confidence, and exclusion counts. Remove utterances containing PHQ,
questionnaire, diagnostic/diagnosis terms, depression-score/label phrases, or
standard PHQ frequency-answer phrases, plus one adjacent utterance on either side.
This conservative heuristic is not proof of absence of leakage or speaker identity.
Retain ordinary symptom discussions. Save audit counts, not transcript text.

Text pilot: utterance-mean 384-dimensional embeddings; text alone and text plus
original audio/facial summaries, with a matched original audio+face control.
Logistic and linear SVM; same 5 outer folds x seeds 42/137/2026, 3 inner folds,
C .01/.1/1, weights none/balanced, inner pooled F1 selection, fixed zero margin.
Fit separate text median/scale/PCA32 and original preprocessing within every fold.

Temporal pilot: 32 equal-duration chronological bins, original 23 acoustic and
49 facial frame means, filtered transcript embeddings assigned by utterance
midpoint. Empty bins masked. Facial frames require success=1, confidence>=.8.
Quality is valid-frame fraction (audio/face), mean supplied confidence (text),
zero for missing bins. Missing confidence uses neutral one, explicitly reported.
Fold-only per-channel standardization; no labels/IDs/demographics as features.
Small per-modality projection to 16 and kernel-3 Conv1d, masked temporal mean,
learned softmax modality gate and linear classification. Matched kernel-1 control
has no cross-bin temporal mixing. Both use the same modalities. Fixed 50 epochs,
Adam lr .001 weight_decay .01, positive weight from fit labels, full-batch CPU,
seeds as above; no early stopping or new hyperparameter search. Train with
random modality dropout (.15) and Gaussian feature noise (probability .3, SD2)
for both ordinary and quality-aware fusion. Preserve at least one modality.
Quality fusion multiplies gate weights by observed mean quality and renormalizes;
availability masks apply to both. Noise quality is divided by five. This is a
controlled known-corruption experiment, not a validated real-world noise detector.

Compare mean/plain, temporal/plain, temporal/quality; also evaluate quality model
with quality replaced by availability (blind ablation). Test clean, audio noise,
facial frame loss (75%), and each missing modality, with identical perturbations
across models. No raw-waveform corruption claims. For each missing modality,
train a matched ordinary temporal control with that modality absent throughout
training, using the same fixed recipe and remaining two modalities. Also report
classical text-, audio-, and face-only controls. The omitted branch remains
masked and has no influence on predictions.

Primary clean screen: +3 F1 percentage points vs matched control, no balanced
accuracy or mean fold AUC loss, recall drop <=5 points, improvement >=2/3 repeats.
Reliability target: clean F1 loss <=2 points versus ordinary temporal model,
mean corrupted-condition F1 gain >=5 points; report every condition, blind
ablation, and seed variation, not just successful comparisons. These are
exploratory screens, not statistical confirmation or clinical performance targets.
Preserve all prior results. Compare old controls on matching folds and cohort;
show prior test results separately. Do not select a model on the inspected test.

Pre-fitting amendment, 2026-10-30: extraction stopped on the first AI facial
timeline exceeding the WAV duration. The prior facial audit confirms exactly two
additional participants with absolute facial/WAV endpoint gaps >1 second (5.66
and 110.953 seconds). Retain the same 161 people, but mask their entire facial
sequence branch. Do not shift, stretch or clip timestamps. The original static
facial-summary controls do not require alignment and remain unchanged. All
temporal/quality/available-modality controls share these masks. The 143 already
successful caches remain numerically valid and retain their original extractor
hash; explicitly whitelist that hash, checking all other cache contract fields.
No classifier had been fitted or evaluated before this amendment.

Label follow-up: check public official documentation; prepare a provider inquiry
without sending correspondence. Provider confirmation and fresh independent
validation remain external requirements, not completed by this experiment.
