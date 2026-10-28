# Label and speech-segmentation investigation

Investigation date: 2026-10-29. Scope: read-only reconciliation of supplied label
files and feasibility of transcript-aligned audio selection. No training, target
replacement or new development/test predictions. Participant-level audit outputs
remain under ignored reports/data_audit/. Stage 10 is still pending.

## Label evidence

All 275 master questionnaire item sums equal the split-file PHQ_Score values.
All 275 master Depression_label values equal split-file PHQ_Binary. Master split
assignments also agree. The older metadata table overlaps 219 participants and
agrees with their split scores and binary labels. Therefore the discrepancies are
present in the supplied files; they were not introduced by model preprocessing.

| Partition | Participants | Released positives | Score >=10 positives | Disagreements |
|---|---:|---:|---:|---:|
| Train | 163 | 37 | 50 | 13 |
| Development | 56 | 12 | 15 | 3 |
| Test | 56 | 17 | 21 | 4 |
| Total | 275 | 66 | 86 | 20 |

All 20 disagreements have released label zero and score at least 10. Their score
distribution is: ten at 10, four at 11, three at 12, and one each at 13, 15 and 16.
No integer score cutoff reproduces all released binary labels. There are no
released positive labels below score 10. The discrepancy is not explained simply
by changing >=10 to >10.

The 219-row Detailed_PHQ8_Labels.csv is internally consistent (item sums match its
totals), but two totals disagree with current split scores. Its existence does
not resolve which questionnaire version should define the target.

Ali et al. explicitly report the same 20 participant IDs and adopt a score-based
correction in their study [2]. We checked the ID sets match exactly without
publishing participant records here. This is independent corroboration, not an
official dataset correction notice. The E-DAIC manual lists the binary and score
fields but does not resolve the mismatch [1]. The older DAIC-WOZ manual explicitly
defines its PHQ8 binary field as score >=10 [3]; it describes a different release.
It is supporting context, not sufficient authority to overwrite E-DAIC labels.

Conclusion: keep the completed released-label experiments intact. A separately
named score>=10 sensitivity study is scientifically defensible if its target is
declared in advance, but it answers a different outcome definition. No automatic
relabeling or promise of better metrics is warranted. A definitive provenance
resolution still requires a release-specific clarification from the provider.

## Interview-condition difference

The manual identifies IDs 300-492 as human-controlled WoZ interviews and IDs
600-718 as AI-controlled interviews [1]. The supplied partitions confirm:

| Partition | WoZ participants (released positives) | AI participants (released positives) |
|---|---:|---:|
| Train | 143 (32) | 20 (5) |
| Development | 46 (10) | 10 (2) |
| Test | 0 (0) | 56 (17) |

Thus training-only stratified CV mostly measures performance on WoZ interviews,
while the original test measures AI interviews. This is a confirmed difference
in recording conditions, not proof it caused any particular prediction error.
Future training-only evaluation should record results by condition. The five AI
positives in training make precise AI-specific estimates difficult. Avoid an
AI-only training recipe or grand claims from this tiny subgroup. Condition/ID
metadata should remain audit metadata, not shortcut predictor inputs.

## Speech evidence and interpretation

All 275 archives contain WAV audio and CSV transcripts with Start_Time, End_Time,
Text and Confidence; none has an explicit speaker column. There are 241 mono
16 kHz recordings and 34 stereo 48 kHz recordings, all with 16-bit samples. Ten
stereo recordings are in training, four in development and twenty in test.
The timestamps can select corresponding feature
frames; they do not themselves identify a speaker. Ali et al. describe the
provided transcripts as participant answers and report transcribing the raw audio
to obtain both participant and interviewer speech [2]. That supports a candidate
participant-speech mask, but the present investigation does not claim manual
speaker validation or a measured diarization accuracy.

The audit merges overlapping nonempty transcript intervals, counts invalid
intervals, inspects WAV headers, and measures coverage on training feature
timestamps. Features outside transcript intervals are **untranscribed**, not
automatically silence or interviewer speech. Zero pitch is likewise not a
silence detector: unvoiced speech can also have zero pitch. No transcript text,
recordings or segment-level personal content is copied into tracked documentation.

Some feature timestamps repeat. The first audit attempt rejected such a sequence;
inspection showed a repeated timestamp rather than backwards time. The audit now
records repeats and gaps and keeps the original values. It does not silently
reconstruct timestamps from row numbers. This is important for later interval
alignment. Existing aggregated tables were not modified.

Across all 163 training sessions, transcript intervals cover a median 60.0% of
feature frames (range 15.0%-99.98%). These are interval-coverage diagnostics, not
verified participant-speech percentages. The median zero-pitch fraction is 57.9%
inside those intervals and 95.6% outside, reinforcing that zero pitch cannot be
used alone to identify silence. There are repeated timestamps in 126 sessions,
one repeat each; the median step is 0.01 seconds, but individual gaps reach 0.26
seconds. Filtering must use actual timestamps, not a reconstructed row clock.

Two training sessions, both from the AI condition, have feature/audio endpoint
mismatches greater than one second. One feature sequence extends 107.04 seconds
beyond its WAV; another stops 485.537 seconds before its WAV ends. Independently
reopened WAV headers agree with the actual archive member byte lengths, so this
is not merely an incorrectly inferred audio duration. The cause (different source
version, truncation, or another extraction issue) is unknown. Endpoint agreement
for the other sessions also does not prove detailed temporal synchronization.
IDs and diagnostics are saved only in the ignored local audit outputs.

Only one transcript interval across all 275 files was invalid or empty under the
audit's checks, in development; none was flagged in training. This validates
structural interval bounds, not transcription completeness or speaker identity.

A further local check streamed all sample frames of the ten stereo training
recordings in bounded chunks. Two had bit-identical channels; the other eight
had channel correlations between 0.9999905 and 0.9999988. These near-identical
signals do not demonstrate usable interviewer/participant channel separation.
No listening-based speaker annotation was performed. Stereo channel diagnostics
are saved separately in training_stereo_channel_audit.csv.

## Recommended next experiment, not executed here

1. Validate a small, fixed sample of transcript-covered and uncovered audio spans
   across both interview conditions. Check speaker identity, timing offsets and
   missing speech before calling the mask participant-only. Keep the review local.
   First resolve the two length anomalies, or explicitly quarantine them from a
   prospective segmentation experiment. Re-extract features from verified source
   WAVs with a documented matching configuration if needed; do not invent offsets
   or silently change the completed experiments' participants.
2. Freeze a comparison of full-session versus transcript-covered eGeMAPS features
   using the same simple classifier and matched nested training folds. Name the
   condition transcript-covered unless speaker checks justify a stronger claim.
   Keep outside-interval and zero-pitch statistics as diagnostics, not assumed
   silence labels. Reuse the actual frameTime values, including duplicates/gaps.
3. Keep the target unchanged for that comparison to isolate segmentation's effect.
   If a score-derived target study is authorized, register it separately and
   report both policies rather than selecting whichever gives higher scores.
4. Report precision, recall, specificity, balanced accuracy and fold AUC alongside
   F1; include simple constant references and condition-specific diagnostics.
   Preserve the already evaluated test set and seek independent confirmation.

This prioritizes a concrete input issue before a larger sequence model. It does
not establish that segmentation will improve prediction.

## Verification and reproduction

Completed audit: `reports/data_audit/label_speech_20261028T232459648689Z` (UTC
directory timestamp; local investigation date 2026-10-29). The folder contains
label reconciliation, legacy questionnaire discrepancies, audio/transcript timing,
the aggregate results and verification files. A separate local JSON records the
two WAV-header/file-size checks. No participant records are committed.

93 tests pass. All 275 archive records and all aggregate timing statistics were
checked. The 163 inspected training eGeMAPS content hashes match the preceding
audio-study extraction. Source/label hashes verified; the previous Stage 9 output
hashes and audio-study input/source hashes remain unchanged.

```powershell
.venv\Scripts\python.exe -m src.dataset.label_speech_audit
.venv\Scripts\python.exe -m pytest --basetemp=.pytest_cache/label-speech-check --tb=short
```

The main audit caches per-participant diagnostics by source hash and archive
size/mtime. It inspects all split labels and transcript/WAV headers, but acoustic
frame diagnostics are restricted to official training participants. The additional
stereo check reads PCM frames in chunks of 65,536, compares left/right sample
equality, and accumulates whole-recording Pearson correlation without exporting
waveforms. These diagnostics do not fit or evaluate a classifier.

## Sources and access limits

[1] USC [E-DAIC manual](https://dcapswoz.ict.usc.edu/wwwedaic/E-DAIC%20Manual.pdf),
accessed 2026-10-29. Its 219-directory description differs from the downloaded
275-participant release. It also describes 88 eGeMAPS measures, whereas our actual
LLD CSV schema contains 23 descriptors plus name/frameTime. Use the verified file
schema for implementation rather than assuming the manual's feature count.

[2] Ali et al., [Leveraging Audio and Text Modalities in Mental Health, v2](https://arxiv.org/html/2412.10417v2),
sections 3.1.1 and 4.4, accessed 2026-10-29. Used for label discrepancy and transcript
provenance evidence only. Its LLM results use a different evaluation setup and are
not benchmark comparisons for this project.

[3] USC [DAIC-WOZ documentation](https://dcapswoz.ict.usc.edu/wp-content/uploads/2022/02/DAICWOZDepression_Documentation.pdf),
accessed 2026-10-29. Different release from E-DAIC.

The official labels directory returned HTTP 403 and the previously referenced
MDPI study returned HTTP 429. No current replacement label package or official
erratum was verified. No provider correspondence was sent, and no dataset files
were downloaded or changed.
