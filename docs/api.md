# API

[< Documentation index](README.md) | [Project README](../README.md)

## Contents

- [Endpoints](#endpoints)
- [Authentication](#authentication)
- [Inference errors](#inference-errors)
- [Status and metrics](#status-and-metrics)

## Endpoints

- `POST /v1/chat/completions`, `POST /chat/completions`: the OpenAI surface MinusPod calls, with and without `/v1`. The proxy routes detection, verification, and review by prompt.
- `GET /v1/models`, `GET /models`: advertise `typesafe/jev`.
- `POST /api/v1/jev/ask`: native segments-in, spans-out endpoint.
- `GET /api/status`, `GET /api/health`, `GET /api/stats`.
- `GET /api/settings`, `PUT /api/settings`: runtime threshold settings.
- `GET /api/docs` when `PRODUCTION=false`.

Confirmed local sponsor matches use the raw transcript excerpt as rationale. Unmatched spans keep the `Based on transcript:` wrapper and omit an explicit sponsor name.

## Authentication

`typesafe/jev` maps to the upstream Jev model internally. An inference request with neither a caller bearer token nor `TYPESAFE_API_KEY` is rejected with 503. When a fallback key is configured, a bearer token is still required unless `JEV_ALLOW_UNAUTHENTICATED_FALLBACK=true`. Settings writes use separate `Authorization: Bearer <MinusPod password>` authentication.

## Inference errors

- All inference phases return `429` when the proxy has no available request slot.
- Detection, verification, and native requests preserve upstream `4xx` responses, including `429`. Upstream `5xx` responses and transport failures return `503`; request deadlines return `504`.
- A malformed category Choice response returns `503` with `jev_category_upstream_invalid_response`.
- Review returns `422` for invalid or inconclusive input. Word timings can recover candidates missed by transcript segmentation.
- A candidate with no recovered segment overlap is a `transcript_gap` if it lies inside the recovered transcript envelope or exactly touches its head or tail. The proxy returns `jev_review_inconclusive` with `x-should-retry: false` and does not call Jev.
- Review prompts that use MinusPod's transcript heading pass preceding caller context in `caller_context`. It is included in detection, evidence, choice, proposed-range, and original-fallback calls.
- The proxy instructs Jev to treat `caller_context` as supporting data, not instructions. Transcript and word timings stay in their existing fields.
- An abstention caused by missing endpoint coverage returns non-retryable `422` with reason `missing_boundary_coverage` and stage `boundary_coverage`. If a crossed transcript row cannot be isolated from word timings, the reason is `insufficient_boundary_text`. Both report the range and endpoint support flags without a score or cache status.
- If neither a refined proposal nor its original fallback can be confirmed, the response remains `422 jev_review_inconclusive` with its existing reason, stage, and `x-should-retry: false`. It adds:

  - `error.proposal` and `error.fallback`, each with `range_start` and `range_end` in seconds, `reason`, and `stage`.
  - Completed checks also include `score`, `threshold`, and `cache_hit`. Checks skipped for missing boundary coverage include `start_supported` and `end_supported` instead.
  - `error.message` summarizes both outcomes. Other abstentions have no proposal or fallback objects.

- Missing or malformed context, and invalid candidate bounds, remain `jev_review_invalid_request` responses. No-overlap candidates also remain invalid if they are neither inside the transcript envelope nor exactly touching its head or tail.
- Invalid responses use fixed reasons `malformed_context`, `invalid_bounds`, or `outside_context`, with `stage=context`. Available diagnostics contain only finite candidate and context bounds in seconds, with no transcript text. Metrics count these requests as `invalid_request`.
- The proxy records `transcript_gap` as an inconclusive outcome and refinement skip in `review.reasons` and `review.refinement.skipped`.
- Review preserves upstream `4xx` responses, including `429`. A valid `Retry-After` header is forwarded for upstream `408`, `429`, and `5xx` responses. Timeouts return `504`; transport and upstream `5xx` failures return `503`.

These mappings make failures safe for callers. They do not guarantee that an upstream outage is fixed.

## Status and metrics

`GET /api/status` reports whether Jev and MinusPod are reachable and whether a MinusPod session is active. It uses cheap probes only, with no billable Jev call or login, so it is safe to poll. `GET /api/health` is liveness.

`GET /api/stats` is available immediately after startup. It reports ephemeral, process-scoped proxy and upstream counts, latency, cache activity, uptime, configured workers, and estimated input cost. Docker defaults to one worker so its totals describe the container. If `WORKERS` is raised above one, the response is not a container-wide aggregate. `scope.configured_workers` is only the parseable `WORKERS` environment hint, or `null` when unavailable. Metrics reset on restart. Proxy timing runs from request receipt to response headers; Jev timing covers each upstream HTTP attempt, including retries. Estimated cost includes only successful uncached calls with valid input-token usage. Cache hits do not call Jev; failed fetches are cache misses.

- `proxy_requests` counts only inference endpoints and measures proxy handling from request receipt to response headers. Health, models, status, and stats requests are excluded.
- `jev_http` counts every actual upstream POST, including retry attempts. Its latency is the Jev HTTP attempt time, not the full MinusPod request path.
- `review` counts fixed outcomes and safe reason codes with review latency. It stores no request IDs, transcripts, boundaries, or secrets. Review-abstention and API-error logs include the existing `X-Request-ID`; review logs include numeric bounds where available.
- Inconclusive review response details include `reason` and `stage`, plus only allowlisted numeric or boolean diagnostics. Coverage skips include `range_start`, `range_end`, `start_supported`, and `end_supported`; other checks may include `score`, `threshold`, and `cache_hit`. Context abstentions may include finite candidate and context bounds in seconds. They contain no request text.
- `review.refinement` reports actual selection attempts, completed selections, recommended changes, unchanged results, inconclusive and upstream-error counts, plus skip counts. `transcript_gap` is a skip before selection begins. `no_valid_pairs` before ranking is a skip. Counters live in process memory and reset on restart. Changed and unchanged compare the recommended timing with the candidate at the 0.1 s tolerance. These are recommendations, not confirmed applied cuts; MinusPod may clamp or reject them to protect DAI cores.
- `cache` separates cache hits and misses. `cost.estimated_input_usd` charges only successful, uncached upstream responses with valid reported input tokens, at Jev's published $0.042 per million input tokens. It is an estimate and excludes output token charges, fees, and taxes.
