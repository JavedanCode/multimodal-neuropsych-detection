# Frozen speech-encoder pilot

Fixed 2026-10-29 before extracting speech embeddings or fitting classifiers.
Distinct follow-up to the supplied DenseNet pilot, which failed its improvement
criterion. No retuning of that completed experiment. Stage 10 remains pending.

Use facebook/wav2vec2-base-960h revision
`22aad52d435eb6dbaf354bdad9b0da84ce7d6156`, local safetensors weights, no remote
code, no cloud inference. This is a LibriSpeech speech-recognition encoder, not a
depression-trained model or the exact model configuration of the competing paper.
Use its frozen last hidden layer, without its CTC prediction head or transcripts.
Model card: https://huggingface.co/facebook/wav2vec2-base-960h

Same 161 training participants and unchanged provisional released labels as the
preceding two pilots. Preserve the two existing timing quarantines in all controls.
Read only training WAVs, average stereo channels, resample to 16 kHz using
scipy.signal.resample_poly, and apply the model's per-window normalization.
Use up to 64 evenly spaced, nonoverlapping eight-second windows per recording,
including first and last available eight-second windows. Count is
min(64,floor(duration/8)); derive integer sample starts by rounding linspace.
No label-, transcript-, pitch- or outcome-based window selection. This bounded
sampling does not cover every moment in long interviews or isolate participant
speech. All windows have equal duration and receive equal aggregation weight.

For each window, mean-pool its last-layer temporal embeddings (768 values).
Across windows compute mean and population SD per dimension (1,536 features).
Save per-window embeddings and starts locally for inspection; never export raw
waveforms. Use batch size four and float32 GPU inference, deterministic seeds,
TF32 disabled, encoder eval mode. Record actual model/source/WAV hashes and GPU
memory/timing. Enforce bounded WAV/member duration and feature shape/finite checks.

Compare original audio; original audio+facial; speech embeddings alone; speech
embeddings+original facial; speech embeddings+temporal facial. Two classifiers
(logistic, linear SVM), plus constant references. Use the identical five outer
folds x seeds 42/137/2026 and three inner folds from previous pilots. Fit all
preprocessing only within fitting folds. Speech branch uses the preceding deep
feature recipe: median imputation, constant removal, standardization, whitened
32-component randomized PCA (seed42; cap to fitting rank dimensions). Feature
names use audio__dense_wav2vec_* solely to reuse that deep-branch compressor;
they are Wav2Vec2 features, not DenseNet features. Face branch preprocessing stays
unchanged. C=[.01,.1,1], weights=None/balanced; select pooled inner F1, first tie;
fixed zero decision margin cutoff, no threshold sweep.

Primary comparisons: speech-only vs original audio; speech+original face vs
original concatenation; speech+temporal face vs original concatenation, within
classifier. Require mean F1 gain >=.03, no balanced-accuracy/AUC loss, recall drop
<=.05, and positive F1 differences in at least two repeats to flag a candidate for
further study. Report all metrics/conditions and uncertainty as repeat SD, not CI.
This exploratory screen is not independent validation or an automatic deployment
decision. Retain all failed conditions. No development/test reevaluation, no label
replacement, and no post-result additional model/hyperparameter search in this pilot.
