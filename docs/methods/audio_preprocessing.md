# Matched waveform preprocessing experiment — 2026-11-08

Freeze before cohort extraction/evaluation. Use the existing 161 retained official
training participants, unchanged released labels and joint domain/label five-fold
splits with seeds 42, 137 and 2026. No development/test evaluation or new tuning.

Compare original supplied audio features with newly extracted eGeMAPSv02 LLD
means and population standard deviations from three waveform conditions:

1. Raw control: existing mono conversion and 16 kHz resampling only.
2. DC removal: subtract the full recording mean.
3. Gentle high-pass: DC removal followed by a second-order Butterworth filter at
   50 Hz, applied forward/backward (zero phase). No amplitude normalization.

All extraction uses openSMILE Python 2.6.0, the same eGeMAPSv02 configuration,
whole recordings and identical summary statistics. Preserve duration and pauses;
do not infer speaker identity, trim speech, suppress estimated noise, or modify
pitch/speed. These are conservative cleanup hypotheses, not confirmed remedies.
Record clipping, DC offset, RMS, duration, source and cache hashes, feature schema,
package version and timings. Cache each participant for safe sequential resumption.

Hold original facial summaries fixed, including the two existing timing
quarantines. Fit audio-only and audio+face for each of the four audio conditions,
plus face-only and constant-class controls. Each fitted model uses fold-only median
imputation, constant/duplicate removal, scaling and balanced logistic regression
C=.1, threshold zero. No inner selection is necessary for this fixed recipe.

Report all 11 conditions: 9 fitted models per fold, 135 saved models total, and
5,313 held-out prediction rows. Report accuracy, balanced accuracy, precision,
recall, F1, fold AUC and AP. Compare each cleanup with the newly extracted raw
control; differences from supplied features also include extractor differences.
The original concatenation must reproduce the previous 37.2% F1 control.

Use the existing paired improvement screen: +3 points F1, no mean balanced-accuracy
or AUC reduction, recall loss at most 5 points, F1 gain in at least 2/3 repeats.
These reused-participant estimates are exploratory, with no multiplicity-adjusted
significance claim. Independently restore saved models, replay the fixed fits,
verify all metrics and compare against the original-feature historical control.
Stop after these conditions; stronger denoising is a separate future hypothesis.
