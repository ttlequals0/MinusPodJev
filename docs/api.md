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
- Review returns `422` for invalid or inconclusive input, including an unknown or low-confidence boundary pair. It preserves upstream `4xx` responses, including `429`. A valid `Retry-After` header is forwarded for upstream `408`, `429`, and `5xx` responses. Timeouts return `504`; transport and upstream `5xx` failures return `503`.

These mappings make failures safe for callers. They do not guarantee that an upstream outage is fixed.

## Status and metrics

`GET /api/status` reports whether Jev and MinusPod are reachable and whether a MinusPod session is active. It uses cheap probes only, with no billable Jev call or login, so it is safe to poll. `GET /api/health` is liveness.

`GET /api/stats` is available immediately after startup. It reports ephemeral, process-scoped proxy and upstream counts, latency, cache activity, uptime, configured workers, and estimated input cost. Docker defaults to one worker so its totals describe the container. If `WORKERS` is raised above one, the response is not a container-wide aggregate. `scope.configured_workers` is only the parseable `WORKERS` environment hint, or `null` when unavailable. Metrics reset on restart. Proxy timing runs from request receipt to response headers; Jev timing covers each upstream HTTP attempt, including retries. Estimated cost includes only successful uncached calls with valid input-token usage. Cache hits do not call Jev; failed fetches are cache misses.

- `proxy_requests` counts only inference endpoints and measures proxy handling from request receipt to response headers. Health, models, status, and stats requests are excluded.
- `jev_http` counts every actual upstream POST, including retry attempts. Its latency is the Jev HTTP attempt time, not the full MinusPod request path.
- `review` counts fixed outcomes and safe reason codes with review latency. It stores no request IDs, transcripts, boundaries, or secrets. Request IDs and numeric bounds are logged for diagnosis.
- `review.refinement` reports actual selection attempts, completed selections, recommended changes, unchanged results, inconclusive and upstream-error counts, plus skip counts. `no_valid_pairs` before ranking is a skip; after ranking it is inconclusive. Counters live in process memory and reset on restart. Changed and unchanged compare the recommended timing with the candidate at the 0.1 s tolerance. These are recommendations, not confirmed applied cuts; MinusPod may clamp or reject them to protect DAI cores.
- `cache` separates cache hits and misses. `cost.estimated_input_usd` charges only successful, uncached upstream responses with valid reported input tokens, at Jev's published $0.042 per million input tokens. It is an estimate and excludes output token charges, fees, and taxes.
