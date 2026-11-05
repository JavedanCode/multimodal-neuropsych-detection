# Estimated speaker-filter sensitivity study — 2026-11-04

Declared before corpus extraction and scoring. The user requests further model
improvement before project completion and authorizes speaker inference from
transcript context. This study does not claim verified diarization. Use only the
same 161 training participants and unchanged released labels. No new dev/test
evaluation, new datasets, label correction, or changes to frozen historical code.

Hypothesis: short interviewer-style prompts in supplied transcripts dilute frozen
MPNet participant summaries. Remove only predeclared prompt patterns, not arbitrary
questions or uncertain answers. Preserve original disclosure screening before
speaker inference; never restore excluded content. Reuse matching original MPNet
utterance embeddings, with source hashes and original-row mapping checked.

Two fixed filters: (1) prompt patterns supported by a nearby subsequent answer;
(2) prompt patterns regardless of nearby answer evidence. Both recognize short
second-person information requests and tell-me-about prompts, max 28 words.
Protect first-person narrative/quotation starts and first-person narrative inside
question-like rows. For contextual support inspect the next three chronological
rows starting within 30 seconds after the prompt ends; require a non-prompt row
with first-person content or a short answer of at most eight words. These are
engineering heuristics, not a learned speaker model or accuracy guarantee.

Recompute per-participant mean embeddings using retained rows. If a participant
would lose all rows, use their original mean and record fallback; never remove a
participant or alter class balance. No quality, IDs, domains, PHQ or demographic
values enter the predictors. All inferred annotations remain separate from the
human listening annotations. Report counts and masks locally and aggregate counts
in tracked documentation. Do not tune rules after looking at outer scores.

Five outer folds jointly stratified by domain/label, seeds42/137/2026, three inner
folds seed+fold+1, identical to baseline_v2. Each representation uses fold-fitted
standardization + whitened PCA32 + linear SVM C .01/.1/1 and weight None/balanced.
Pure reference, context-filter, prompt-filter policies select C/weight by pooled
inner F1 at cutoff0. Report all. Primary selected policy includes reference plus
both filtered variants and chooses pooled inner F1, reference first on ties.

Additional precision controls: threshold-only selects cutoff0/.1/.2/.3 for the
inner-selected original reference setting; joint precision policy searches those
cutoffs across all three representations/settings. Eligibility requires recall>=65%
and F1>=original reference inner F1-.02; maximize precision,F1,recall, first tie;
if no candidate is eligible, retain reference. All decisions use inner labels only.

Six reported conditions: reference, threshold, context_filter, prompt_filter,
selected, precision. Primary clean F1 screen +3pp with no BA/AUC drop, recall
loss<=5pp and F1 gain in >=2 repeats. Precision screen +5pp, recall>=65%, recall
loss<=5pp, F1 loss<=2pp and precision gain in >=2 repeats. Neither the best outer
seed nor the most favorable metric can redefine success. Report AP, domains,
confusion counts and paired conditional participant bootstrap as in bounded_v3.

Stop this series after these conditions; use results to decide whether estimated
speaker filtering warrants further work. A negative result preserves the baseline.
This addresses a new input hypothesis supported by the pilot, not an extension
of the completed bounded_v3 parameter grid. All scores remain exploratory because
these participants have informed previous research decisions.
