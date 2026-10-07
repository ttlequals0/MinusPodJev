# Changelog

## 0.1.30 - 2026-10-06

- Add two new runtime-editable review settings: boundary candidate cap (seconds) and context window (seconds), both with validation and startup environment defaults.
- Return the original interval when a proposed cut fails closing-phrase validation but the original passes all absolute checks. Include ranges and scores in diagnostics when both fail with new reason `terminal_closing_unconfirmed`.
- Read legacy three-field and four-field settings files, fill the missing fields from startup defaults, and write six fields on the next save.
- Build the frontend stage on the build platform so arm64 images no longer run node under emulation.
- Keep the Jev response cache on the data volume in Compose with `JEV_CACHE_PATH=/app/data/jev_cache.json`. Document where the model is set in MinusPod, the login password versus the master passphrase, a Podman Quadlet unit, and the current state of Jev as a reviewer.

## 0.1.29 - 2026-09-28

- Let a complete closing offer, URL, or sign-off at the end of supplied review speech define the final ad utterance. Treat truncated endings as inconclusive; missing later speech alone does not prove the audio ends.

## 0.1.28 - 2026-09-28

- Return the canonical name for a known sponsor in detection output so downstream brand checks can match spoken mentions. Unknown detection spans still omit the sponsor field.

## 0.1.27 - 2026-09-28

- Ground review evidence in the candidate's observed words when boundary refinement can validate the full interval. Check added start speech for a spoken return to the episode before widening a cut. Keep the existing evidence check for reviews without usable word timing.

## 0.1.26 - 2026-09-28

- Include consecutive removable ads and produced trailers in one review cut. Preserve programme and configured keep speech, and validate an aligned earlier end if extending the cut fails safety checks.
- Apply a programme or configured keep speech veto to any larger review interval that contains the protected interval.

## 0.1.25 - 2026-09-27

- Anchor review boundaries to the configured removable category run that intersects the candidate. Apply explicit category actions to boundary and safety checks, including kept-category protection.
- Treat podcast trailers and story teasers that lead to a follow, listen, or subscribe call to action as promotional speech.
- Compare a selected start with one earlier observed boundary when the transcript may contain an omitted promotional setup. Show both sides of fine word boundaries so opening fragments stay with the promotion.
- Center coarse end context on a corroborated advertising edge within the configured search cap while retaining the original candidate and supplied boundary options.
- Preserve same-show follow, subscription, and early-access offers when self-promotion is configured to stay.
- Compare the selected closing utterance with the next one before placing the final word boundary. This keeps a late sponsor URL inside the cut while leaving the return to the show intact.

## 0.1.24 - 2026-09-27

- Use supplied word boundaries up to 60 seconds from each review edge. Fine word ranking gets 30 seconds of context centered on the selected coarse boundary.
- Treat an explicit `neither` interval comparison as inconclusive instead of returning a separately validated interval.
- Compare intervals only when both have complete transcript and boundary support. A sole supported range still must pass every absolute safety check.
- Give programme-speech checks the complete candidate passage. This connects staged commercial scenes and sponsor-linked personal setups to their payoff without reclassifying an independent story.

## 0.1.23 - 2026-09-27

- Allow a complete sponsor closing phrase at the end of the supplied review context to define the final word boundary. Continue to abstain when the context ends mid-URL or mid-phrase.

## 0.1.22 - 2026-09-26

- Honor the provider's explicit Choice selection even when another option has a higher independent score. Existing interval safety checks still control review adjustments.
- Preserve positive-duration transcript lines that round to a point timestamp, while rejecting reversed intervals.
- Compare neighboring utterances at both boundaries and the removed and kept speech at the end, then check whether the cut includes the opening of the kept programme phrase.
- Keep generic hooks and commentary in a sponsor read when nearby speech ties them to its problem, claim, product, or offer.

## 0.1.21 - 2026-09-26

- Select the utterance that contains the final sponsor word before choosing its exact word boundary. Check nearby excluded speech for sponsor continuity. Check the final four words and interior sentence pairs for programme speech.
- Prefer complete produced ad scenes and full URLs or sign-offs. Keep the original interval available as a safety fallback.
- Apply the configured review evidence threshold during the review prefilter. Standalone detection thresholds remain unchanged.

## 0.1.20 - 2026-09-26

- Exclude reviewer cut boundaries that split a supplied word. Keep supported word and silence boundaries eligible.

## 0.1.19 - 2026-09-26

- Rank observed utterance starts and refine the selected utterance to a word boundary. Keep the original cut available when the proposed start is uncertain.
- Confirm sponsor speech and check for programme speech in each eligible interval. The programme veto uses a separate `JEV_REVIEW_PROGRAMME_VETO` threshold, defaulting to 0.85.
- Add an opt-in probe for archived reviewer inputs.

## 0.1.18 - 2026-09-25

- Check each eligible review cut for ad-only speech independently. Use the configured review threshold for that check, then compare two safe cuts to choose the boundary.
- Keep a supported ad-only cut when the other cut lacks transcript coverage or contains show speech. Report ad-content uncertainty separately from relative boundary preference.

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
