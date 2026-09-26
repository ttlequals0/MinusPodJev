# Changelog

## 0.1.17 - 2026-09-25

- Offer every supplied word-timed start and end within 30 seconds as separate boundary choices. Abstain if either choice would exceed Jev's option limit.
- Compare the proposed complete cut with the original in one final Choice that can reject both. Keep the configured review threshold and transcript coverage checks.
- Omit duplicate word arrays from the advertising-evidence request while retaining line timestamps. Detection behavior is unchanged.

## 0.1.16 - 2026-09-25

- Use bounded transcript context for boundary choices instead of repeating full word arrays. Log selected timestamps and safe request sizes on upstream review errors.
- Check the selected cut for unrelated show speech and whether adjacent words continue the same promotional message. Apply these checks to the original-cut fallback as well as changed boundaries.
- Keep review thresholds and detection behavior unchanged. Boundary refinement remains experimental and needs live accuracy testing.

## 0.1.15 - 2026-09-25

- Offer Jev word-timed boundary choices with nearby speech and include unpunctuated word edges close to the current cut.
- Validate only the selected speech, then check any speech added to or removed from the original cut before accepting a boundary change.
- Abstain when a partial transcript row cannot be reconstructed from word timing, and report `insufficient_boundary_text` separately from missing endpoint coverage.

## 0.1.14 - 2026-09-25

- Read only the transcript block in detection requests so timestamped show notes cannot override segment IDs.
- Redact personal email addresses from benchmark artifacts.

## 0.1.13 - 2026-09-24

- Forward the review prompt preamble as separate `caller_context` data through Jev review calls.
- Abstain with non-retryable `422` and no Jev call for transcript gaps inside the recovered envelope or exactly at its head or tail.
- Return fixed context error reasons and finite candidate/context bounds without transcript text.

## 0.1.12 - 2026-09-23

- Send the selected review cut as an explicit `assessment_range` while retaining the original candidate range for reference.
- Clarify that timestamp gaps alone do not prove a pause, silence, or content absent from the transcript.
- Return separately sanitized proposal and fallback diagnostics when neither the proposed cut nor original fallback can be confirmed.
- Keep review thresholds and cache behavior unchanged. Boundary refinement remains experimental. Its accuracy is unproven.

## 0.1.11 - 2026-09-23

- Fix the original endpoint gate so supported corrections can reach boundary review.
- Account for partial or zero-point word timing without inventing transcript text.
- Report safe `missing_boundary_coverage` diagnostics and metrics while retaining independent complete-range validation and original-cut fallback.
- Keep review thresholds and cache behavior unchanged.
- Changes tracked in [PR #7](https://github.com/ttlequals0/MinusPodJev/pull/7), covering boundary search, sponsor learning, transcript-gap recovery, and review-gate corrections.

## 0.1.10 - 2026-09-23

- Recover valid ad candidates when supplied word timings expose a gap in transcript segmentation. Candidates still inside the supplied context with no segment overlap return `422 jev_review_inconclusive` without Jev calls.
- Rank inward boundary choices, then validate the proposed complete cut with a focused NouL. If the adjustment is uncertain, confirm the original cut only after validating it independently.
- Keep review evidence and boundary validation thresholds unchanged. Cache behavior is unchanged.
- Include safe reason, stage, and available score, threshold, and cache-hit diagnostics in inconclusive review responses and logs, without request text.

## 0.1.9 - 2026-09-23

- Return `422 jev_review_inconclusive` when a valid candidate inside the coarse context overlaps no segment. Skip Jev calls and boundary-selection attempts, and count the inconclusive outcome, reason, and skip.
- Include the existing `X-Request-ID` in review-abstention and API-error logs.

## 0.1.8 - 2026-09-23

- Search inward boundary trims at 2-second steps up to 30 seconds, snapping to supplied word timestamps.
- Rank start and end boundaries in one Jev request, then choose a pair from the strongest candidates or keep the current pair. Unknown or low-confidence results return 422.
- Use the raw transcript excerpt for confirmed local sponsor matches, avoiding MinusPod's generated-rationale rejection.
- Clarify that boundary selections are recommendations; MinusPod controls cuts and DAI protection.
- Count empty boundary searches as skipped before ranking, with an explicit `no_valid_pairs` reason.

## 0.1.7 - 2026-09-22

- Preserve upstream status and retry metadata when review requests fail.

## 0.1.6 - 2026-09-21

- Classify each detected span with one validated Jev Choice across MinusPod's category labels.

## 0.1.5 - 2026-09-21

- Made all four runtime thresholds persistent and editable, including detection stay.
- Replaced independent boundary-word choices with one constrained boundary-pair choice.
- Accept complete Choice probability maps totaling 0.99 to 1.01 without changing supplied confidence.
- Return sanitized upstream errors for detection and native requests instead of generic 500 responses.

## 0.1.4 - 2026-09-20

- Skip end-boundary selection after an inconclusive start.
- Log review-validation and settings-file errors safely; reject malformed review usage.
- Changes tracked in [PR #2](https://github.com/ttlequals0/MinusPodJev/pull/2).

## 0.1.3 - 2026-09-20

- Added persistent runtime threshold settings with authenticated status-page editing.
- Fail closed on malformed saved settings instead of silently resetting to defaults.
- Added MinusPodJev logo, favicon, and status-page branding.
- Changes tracked in [PR #2](https://github.com/ttlequals0/MinusPodJev/pull/2).

## 0.1.2 - 2026-09-20

- Added effective review settings and thresholds to `/api/status`.
- Added refinement counters and skip reasons to `/api/stats`.
- Added precise review-stage, Choice-confidence, and boundary-result logs.
- Kept review thresholds and behavior unchanged; refinement remains opt-in.
- Changes tracked in [PR #2](https://github.com/ttlequals0/MinusPodJev/pull/2).

## 0.1.1 - 2026-09-20

- Fixed review handling, kept word refinement opt-in, and added review metrics.
- Switched to Alpine and a locked runtime virtual environment; removed package installers from the final image.
- Changes tracked in [PR #2](https://github.com/ttlequals0/MinusPodJev/pull/2).
