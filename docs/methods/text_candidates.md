<!-- Original frozen protocol: 5f3e3f3c4f754c25e92f377cdfedcf822d561a71:docs/bounded_v3_protocol.md -->
# Bounded existing-data study — 2026-11-04

Freeze before feature generation or candidate scoring. No new dataset is available.
Official development and test participants have already been evaluated in earlier
stages; neither can be called untouched. Use only the existing 161 training
participants and unchanged released binary labels. Keep speaker status unresolved
until human evidence exists. No re-labeling, re-splitting to improve scores, new
dev/test evaluation, encoder fine-tuning, or deployment in this study.

Six reported procedures: original reference; reference with precision threshold;
topic-balanced text; concatenated topic text; conservative quality residual fusion;
inner-selected candidate including reference. Fixed seeds 42/137/2026 and joint
domain/label five outer folds, three inner folds, matching baseline_v2. Every
scaler/PCA/model/threshold/selection fits only the appropriate training subset.

Fixed semantic anchors, encoded by the existing pinned frozen MPNet encoder:
daily activities and hobbies; friends and family; work and education; sleep and
energy; feelings and coping; physical health and exercise. Assign each already
screened utterance to its highest-cosine anchor. These are automatic topic proxies,
not clinically validated categories or speaker labels. Empty topics use the
participant's global mean. Balanced text averages the six topic means; concatenated
text concatenates them. No outcomes or cohort-fitted clustering define topics.
Both use training-fitted standardized whitened PCA32 and linear SVM, C .01/.1/1,
weights None/balanced. The global mean reference is the original MPNet table.

Residual fusion adds at most alpha=.1 or .25 times an auxiliary audio/face margin
to the original text margin; alpha=0 is included. Auxiliaries are standardized
logistic C=.1 balanced. Divide auxiliary margins by their fitting-set margin
standard deviation (minimum 1e-6); no label-dependent calibration of quality.
For audio, availability AND mean clipping<=.001 AND zero fraction<=.01 AND
coverage>=.95 define eligibility. Face eligibility is availability AND mean valid
frame fraction>=.8 AND mean confidence>=.8. These fixed engineering thresholds
are hypotheses, not measured reliability probabilities. Average eligible auxiliary
margins, divide by max(1,eligible count); fall back exactly to text if none.

Reference recipe selects its six settings by pooled inner F1, cutoff0, first tie.
Every candidate setting uses cutoffs 0/.1/.2/.3. Precision selection requires
pooled inner recall>=.65 and F1>=reference inner F1-.02, then maximizes precision,
F1, recall in that order, first tie. Reference itself is a candidate; if none is
eligible, fall back to reference and record the unmet inner constraint. Threshold
condition uses only the selected reference setting. Topic conditions include the
reference fallback. Residual uses the selected reference setting and the three
alpha choices. Selected searches the union, never outer outcomes. Freeze this
finite grid; no extra candidates based on results.

Report all procedures on clean, missing_audio, missing_face, missing_both. Text
procedures are unchanged in these scenarios. Do not supply corruption severity.
Primary precision screen: +5pp precision, recall>=65%, recall loss<=5pp, F1 loss
<=2pp, positive precision difference in >=2 repeats. Also report original clean
F1 screen (+3pp, no BA/AUC loss, recall loss<=5pp, gain in >=2 repeats). No
post-hoc metric switch to declare success. Include AP, domains, confusion counts,
all inner choices, hashes and restored-prediction verification. Stop searching
after this series, whether positive or negative.

Uncertainty: paired bootstrap resamples participants within domain/label strata,
carrying all three repeat predictions together, 2000 draws, seed20261104. Report
95% percentile intervals for mean precision/F1 differences. These are conditional
descriptive intervals for existing cross-fitted predictions; they omit retraining
and adaptive model-search uncertainty and are not independent confirmation.

Speaker-verified features are deferred, not inferred from transcript intervals.
The 24-clip human worksheet is a pilot check, not validation of all recordings.
Existing data can support exploratory evidence; no guarantee of zero overfitting
or proof of the globally best model is possible.
