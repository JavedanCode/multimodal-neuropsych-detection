# Provider inquiry draft (not sent)

Public-documentation check: 2026-10-30. The official E-DAIC manual lists
PHQ_Binary and PHQ_Score but does not specify how to reconcile their disagreement.
No verified official correction was found in that manual. Existing local audit
remains the evidence for the 20 discrepancies; labels have not been modified.
https://dcapswoz.ict.usc.edu/wwwedaic/E-DAIC%20Manual.pdf

Subject: Clarification of E-DAIC released binary labels and transcript provenance

We are conducting participant-level depression-screening research using the
released E-DAIC training, development, and test splits. In our 275-participant
copy, 20 participants have PHQ_Binary=0 and PHQ_Score>=10 (13 train, 3 development,
4 test). The master binary labels agree with the split labels, and questionnaire
item sums agree with the split scores. No integer score cutoff reproduces all
released binary labels.

Could you confirm the intended definition of PHQ_Binary, the applicable release
version, and whether an official erratum or replacement label file exists?
Should released-label and score-threshold experiments be treated as distinct
targets? We can provide participant IDs through your approved private channel.

Could you also confirm whether the supplied Start_Time/End_Time/Text/Confidence
transcripts contain only participant speech, how confidence is defined, and
whether timestamps share the time origin of the supplied audio and OpenFace
features? Two training recordings have substantial WAV/feature endpoint
disagreement; we have quarantined them pending clarification.

Until clarified, we retain the unchanged released labels and explicitly report
the uncertainty. Thank you.

No email or other external message has been sent. This draft requires the
researcher's dataset access account/contact route and explicit sending instruction.
