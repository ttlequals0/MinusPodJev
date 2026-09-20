# Changelog

## 0.1.4 - 2026-09-20

- Skip end-boundary selection when the start boundary is inconclusive.
- Add safe review-validation and settings-storage diagnostics; reject malformed review usage.
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
