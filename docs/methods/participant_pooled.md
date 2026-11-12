# Participant speech, pooled cohort, text-assisted fusion and abstention — 2026-11-12

Freeze before fitting any classifier. Requested by the user on 2026-11-12,
including adding development participants to cross-validation. Released
`PHQ_Binary` labels are unchanged. The test split is not read. Fixed recipe
everywhere: BranchPreprocessor (fold-only imputation, constant/duplicate removal,
scaling) and balanced L2 logistic regression, liblinear, C=.1, all features,
margin threshold zero. No threshold, C, or feature search. Stop after this run.

## Participant-speech separation (label-free)

Source: supplied `OpenSMILE2.3.0_egemaps.csv` descriptors (10 ms frames). Voiced
frames (F0 > 0) are grouped into segments (gaps up to 0.2 s merged, at least
0.2 s voiced). Each segment is described by medians of pitch, HNR, MFCC1-4,
alpha ratio, Hammarberg index, jitter, shimmer and loudness relative to the
session median. The virtual interviewer has one voice per interview domain, so
a diagonal-Gaussian interviewer template per domain is learned from sessions
whose median segment pitch is below 27 semitones, using their segments with
pitch 36-45 semitones. Each session's segments are then assigned by likelihood
to the fixed template or a session-specific participant Gaussian (up to 10
iterations). A domain with fewer than 150 anchor segments uses a template
pooled over both domains; if no template exists or a session has too few
segments, no segment is removed. (A first label-free pass required 300 anchor
segments per domain and found none: interviewer-range segments are rare, about
1% of voiced segments in low-pitch sessions, so the requirement was lowered
before any classifier was fitted.) These constants were set on acoustic and transcript
evidence only, before any classifier was fitted; no questionnaire label was read.

Proxy accuracy (not ground truth): transcript rows whose normalized text recurs
in at least 15 sessions (scripted prompts) should be assigned to the
interviewer; rows of 10+ words that occur once should be assigned to the
participant. Agreement is reported per domain. Transcripts never influence the
assignment.

Two matched audio tables (49 features each: 23 descriptors x mean/std over
voiced frames, voiced fraction, segments per minute, mean segment length):
`segment` (all segments) and `participant` (interviewer segments removed).

## Cohorts

- **T161**: the existing 161 training participants and the existing folds
  (5 outer, seeds 42/137/2026). Directly comparable with earlier runs.
- **P217**: T161 plus all 56 development participants, folds rebuilt by the same
  rule and seeds. The development split has been inspected in earlier work, so
  it is not a holdout. Development facial timing and audio length have not had
  the training cohort's anomaly checks; no development participant is excluded.

## Policies

T161: `original_audio`, `voiced_audio` (earlier 47-feature table),
`segment_audio`, `participant_audio`, `face`, `original_concat` (control, must
reproduce history), `voiced_concat`, `segment_concat`, `participant_concat`,
`original_concat_nopose`, `participant_concat_nopose` (drop the six
`facial__pose_T*` head-translation summaries, which mainly encode camera
geometry), `text` (the retained MPNet reference recipe, unchanged),
`text_av_equal` and `text_av_weighted`: common-scale late fusion of the text
branch and the `original_concat` branch (margins divided by the RMS of inner
out-of-fold margins, where the inner text margins refit the parameters that the
text branch's own inner selection chose on the same outer-training participants;
text weight .5, or chosen from [.5, .25, .75] by pooled
inner-OOF F1, first on ties). Text-assisted policies are a labelled variant, not
a replacement for the audio/face-primary design.

P217 (suffix `_p217`): `original_audio`, `segment_audio`, `participant_audio`,
`face`, `original_concat`, `segment_concat`, `participant_concat`. Reported on
all 217 and restricted to the 161 training participants.

## Comparisons

Existing screen unchanged (mean F1 +3 points, no balanced-accuracy or fold-AUC
loss, recall loss <= 5 points, F1 gain in >= 2/3 repeats); engineering criteria.

- Against `original_concat` (T161): `participant_concat`, `segment_concat`,
  `original_concat_nopose`, `participant_concat_nopose`.
- Isolating: `participant_audio` vs `segment_audio`; `participant_concat` vs
  `segment_concat`; `participant_concat_nopose` vs `participant_concat`.
- Text-assisted: `text_av_equal` and `text_av_weighted` vs `text`.
- P217: `participant_concat_p217` and `segment_concat_p217` vs
  `original_concat_p217`.
- More data (descriptive, different folds): each `_p217` policy restricted to
  the 161 training participants against its T161 counterpart.

## Abstention analysis (descriptive, no fitting)

For `original_concat`, `participant_concat`, `text` and `text_av_weighted` on
T161: rank each repeat's held-out predictions by |score| and report accuracy,
balanced accuracy and F1 on the most confident 100/80/60/40%. Coverage cut-offs
use held-out scores, so this describes whether confidence is informative; it is
not a deployable threshold. Also report `original_concat` separately for
participants flagged by the fixed audio/face quality rules and those not flagged.

Report every result, including failures, with always-positive F1, AUC and AP.
All results are exploratory on a reused cohort.

## Pre-fit observations (label-free, recorded before fitting)

- Seven unambiguous scripted interviewer prompts occur in 164 transcript rows
  across train and development sessions. In 90% of those rows the supplied
  descriptors contain no voiced segment: the interviewer's voice is usually too
  quiet in the participant's recording to register as voiced. Interviewer speech
  is therefore a minor part of voiced-frame summaries, and a small effect of
  separation is expected.
- The separation assigns a median 3.6% of voiced segments to the interviewer
  (10th-90th percentile 0.7%-11.9%, maximum 33%); 87 of 219 sessions exceed 5%.
- The automatic "recurs in 15+ sessions" prompt proxy also matches participant
  phrases such as "i don't know", so its scripted-prompt agreement (13% WoZ,
  50% AI, 76 rows) is not a valid accuracy estimate. Long unique rows are
  assigned to the participant in 98-99% of 7,829 rows. Separation accuracy for
  higher-pitched participants is unverified and may remove some participant speech.
