# Transcript-context speaker estimates — 2026-11-04

The user authorized inferring speaker roles from transcript context. This pilot
covers the 11 previously flagged utterances in six training participants. Original
listening annotations, labels, trained models and feature caches are preserved.

Source transcripts were checked against existing MPNet transcript hashes, ordered
by actual timestamps, and matched to the full target interval. Each case was read
with up to three preceding and following transcript rows. Source row indices and
timestamps are retained. No depression outcomes or classifier predictions were
used to assign roles. Raw text and participant-level estimates remain in ignored
local `data/processed/transcript_context_v4/` artifacts.

## Estimates

| Estimated role | Cases | Evidence |
|---|---:|---|
| Participant | 7 | Personal accounts, sentence continuations, or answers to surrounding questions |
| Participant, tentative | 1 | Standalone negative response among other brief answers; prompting question absent |
| Interviewer | 2 | Direct second-person questions followed by participant-style answers |
| Interviewer, moderate support | 1 | Second-person question fragment among similar prompts; incomplete context |

All three initial human interviewer flags are supported by the text-based
inference. They concern a family question, a dream-job question, and a question
fragment. This is converging evidence about these particular spans, not measured
speaker-recognition accuracy or an estimate of dataset-wide contamination.

Some neighboring transcript rows also appear to contain a prompt and an answer
within the same row. Row-level filtering therefore cannot guarantee speaker-clean
features. The presence of a question mark, or a second-person pronoun alone, is
not a sufficient universal exclusion rule. Participants can ask questions, quote
other people, or use generic second-person language in personal explanations.

Estimates are saved in `role_estimates.json` with per-case reasons, qualitative
evidence strength, source-row mapping and context hashes. Strength is not a
calibrated probability. The weak standalone response is retained provisionally.
The three estimated interviewer spans are marked as candidates for exclusion in
a separately declared sensitivity study, not automatically removed from historical
training features. A wider contextual labeling pass is needed before applying a
speaker filter throughout the cohort. These 11 pilot labels cannot be extrapolated
to all utterances or used to claim the speaker issue is solved.

Recreate the chronological context with:

```powershell
.venv\Scripts\python.exe -m src.dataset.transcript_context_v4
```

The estimates are assistant-authored contextual annotations; the extractor does
not itself perform speaker classification. They remain separate from human
`annotations.json` and any subsequent `annotations_context.json` review.
