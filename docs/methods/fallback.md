<!-- Original frozen protocol: 5f3e3f3c4f754c25e92f377cdfedcf822d561a71:docs/primary_fallback_v5_protocol.md -->
# Audio/facial primary model with text fallback — 2026-11-04

Freeze before fitting. User direction: improve performance before project completion,
keeping actual acoustic and facial measurements primary and text as a fallback.
Use the existing 161-person training cohort and provisional released labels. No
new dev/test evaluation, changed labels, new datasets, or speaker-clean claims.
Supplied acoustic measurements derive from recordings; transcript timestamps still
define the existing aligned intervals. Thus primary classification is text-feature
free, but preprocessing is not independent of transcript timing.

Same five outer/three inner joint domain/label folds and seeds42/137/2026 as v2.
Use existing aligned audio/facial summaries and measured quality. Base predictors:
standardized balanced logistic C=.1 per auxiliary modality. Text fallback: fixed
MPNet PCA32 balanced linear SVM C=.1, cutoff0, fit on training participants only.
Also report the retained fold-selected original text reference separately.

Primary fusion uses audio/face margins and their availability, never text margins,
text embeddings, ASR confidence, labels, IDs or demographics. Quality version adds
audio-margin x six audio-quality and face-margin x three face-quality interactions.
Base margins used to train the fusion learner are out of fold inside each fitting
set; repeat this cross-fitting inside the inner model-selection folds. Meta logistic
C=.01/.1/1, balanced, cutoff0. Train with clean/missing-audio/missing-face views,
each weighted 1/3. Select C per evaluated procedure by pooled inner clean F1.

Plain primary masks unavailable modalities. Quality primary additionally applies
the fixed v3 eligibility rules: audio coverage>=.95, clipping<=.001 and zero
fraction<=.01; face valid fraction>=.8 and confidence>=.8; availability required.
These are pre-existing engineering quality proxies, not calibrated reliability.

Six conditions: primary_plain, primary_quality, availability_fallback,
quality_fallback, fixed_text, retained_reference. Availability fallback uses text
only when neither auxiliary is available; quality fallback only when neither
auxiliary meets the fixed quality eligibility rule. No prediction correctness,
label, or supplied corruption severity may trigger fallback. One usable auxiliary
keeps primary inference active. Text is assumed available for this experiment;
raise an error if a required fallback lacks text rather than inventing an answer.

Evaluate clean, missing_audio, missing_face, missing_both. Save route decisions
and routing fractions alongside scores, metrics, domains and AP. Primary-only
controls without any available auxiliary reduce to their learned intercept; those
are ablation controls, not meaningful predictions from observed evidence. Do not
interpret gains under total input absence as proof of quality-signal usefulness.

Report paired comparisons to retained_reference and to the corresponding primary
control. Clean promotion screen remains +3pp F1, no BA/AUC loss, recall loss<=5pp,
F1 gain in >=2 repeats. Fallback benefit is descriptive by scenario, including
route frequency, not a post-hoc winner criterion. Report all candidates and stop
this fixed series after verification. Previous negative studies remain intact.
