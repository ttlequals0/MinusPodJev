# Environment variables

[< Documentation](README.md) | [Project README](../README.md)

---

## Contents

- [Deployment notes](#deployment-notes)
- [Standard](#standard)
- [Security and access](#security-and-access)
- [Thresholds and review](#thresholds-and-review)
- [Sponsor lookup](#sponsor-lookup)
- [Server and diagnostics](#server-and-diagnostics)

## Deployment notes

The proxy reads these settings from the process environment. A direct source run also reads a
`.env` file in its working directory. Docker Compose uses `.env` for interpolation, but only
variables listed under `environment:` are passed into the container. The Compose file currently
passes the MinusPod settings, all four threshold settings, review flags, fallback flag, and settings
path. It does not pass every variable in this reference; for example, `TYPESAFE_API_KEY` is not
passed by the provided Compose file.

The image serves nginx on container port `8080` and supervises `WORKERS` Uvicorn workers on
`0.0.0.0:8000` (one by default). Compose publishes nginx on `127.0.0.1` by default. Set
`JEVPROXY_BIND_ADDRESS=0.0.0.0` only when an intentional LAN or reverse-proxy exposure has its
own access controls.

## Standard

| Variable | Default | Description |
|----------|---------|-------------|
| `TYPESAFE_API_URL` | `https://api.typesafe.ai/v1/systemone` | TypeSafe Jev System One endpoint. |
| `JEV_MODEL` | `jev-latest` | Model name sent upstream. |
| `JEV_PUBLIC_MODEL` | `typesafe/jev` | Model ID advertised by the proxy. |
| `TYPESAFE_API_KEY` | _(none)_ | Fallback key. It works when explicitly passed to any process, including a container. Normal callers send the TypeSafe key as their `Authorization: Bearer` token. |
| `JEV_TIMEOUT_SECONDS` | `60` | Upstream HTTP timeout in seconds. |
| `JEV_MAX_RETRIES` | `2` | Retries for transient upstream failures. |
| `JEV_RETRY_AFTER_MAX_SECONDS` | `5` | Maximum upstream `Retry-After` delay in seconds. |
| `JEV_REQUEST_DEADLINE_SECONDS` | `75` | Total cooperative budget for one Jev operation, including retries. Keep it below nginx's 90-second timeout. |
| `JEV_MAX_CONCURRENT_REQUESTS` | `4` | Maximum simultaneous upstream requests per worker. |
| `JEV_CACHE_PATH` | `./jev_cache.json` | Legacy JSON cache import path. Active SQLite cache data is stored beside it. |
| `JEV_CACHE_MAX_ENTRIES` | `10000` | Maximum cached response entries. |

## Security and access

| Variable | Default | Description |
|----------|---------|-------------|
| `JEV_ALLOW_UNAUTHENTICATED_FALLBACK` | `false` | Permit `TYPESAFE_API_KEY` when a caller omits bearer auth. Use only for a protected internal deployment. A caller bearer token is otherwise required. |
| `MINUSPOD_PASSWORD` | _(none)_ | MinusPod login password used for the sponsor API and for authenticating proxy settings writes. This is not the TypeSafe API key. |
| `MINUSPOD_BASE_URL` | _(none)_ | MinusPod base URL. When unset, sponsor naming uses the built-in gazetteer. |
| `CORS_ORIGINS` | `["http://localhost:3000", "http://localhost:5173"]` | JSON list of allowed browser origins. |

## Thresholds and review

Values are probabilities from `0` to `1`. `JEV_ENTER` must be at least `JEV_STAY`.

| Variable | Default | Description |
|----------|---------|-------------|
| `JEV_ENTER` | `0.95` | Opens a detected span. Compose passes this startup default. |
| `JEV_STAY` | `0.40` | Extends an open span. Compose passes this startup default. |
| `JEV_REVIEW_EVIDENCE_THRESHOLD` | _(unset, inherits `JEV_ENTER`)_ | Evidence threshold for review. Blank values inherit `JEV_ENTER`. |
| `JEV_REVIEW_CHOICE_THRESHOLD` | _(unset, inherits `JEV_ENTER`)_ | Focused NouL score threshold for selected speech and changed-edge checks. Blank values inherit `JEV_ENTER`. |
| `JEV_REVIEW_REFINE_BOUNDARIES` | `false` | Use Jev Choice questions and supplied word timings to refine review boundaries. |
| `JEV_SETTINGS_PATH` | `./data/runtime-settings.json` | Runtime threshold file. Compose passes this variable and separately mounts the default `/app/data` directory to the `jevproxy-data` volume. |

`GET /api/settings` and `PUT /api/settings` expose detection enter, detection stay, review evidence,
and review Choice values.

- A saved four-field file overrides environment defaults until it is changed or removed.
- A legacy three-field file uses the startup `JEV_STAY` value until its next save writes all four fields.
- The saved file survives container replacement only when its containing directory remains persistent.
- The default path is under `/app/data`, which Compose mounts to a named volume. A custom path must
  also be covered by a persistent mount.
- The Compose volume must be writable by the app user (UID/GID `1000`).
- Runtime changes apply to new requests; an in-flight request keeps its threshold snapshot.

## Sponsor lookup

| Variable | Default (seconds) | Description |
|----------|---------|-------------|
| `SPONSOR_CACHE_TTL_SECONDS` | `3600` | Lifetime of the cached MinusPod sponsor matcher. |
| `MINUSPOD_SESSION_TTL_SECONDS` | `1800` | How long to reuse a cached MinusPod login session. |
| `SPONSOR_FAILURE_COOLDOWN_SECONDS` | `900` | Per-worker delay before retrying a failed sponsor refresh. Account for worker and replica counts against MinusPod login limits. |

## Server and diagnostics

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Backend settings default. The supervised image invokes Uvicorn with `0.0.0.0:8000` explicitly. |
| `PORT` | `8000` | Backend settings default. The image also declares `PORT=8080`, but the supervisor command uses port `8000`; nginx serves the container on `8080`. |
| `WORKERS` | `1` | Worker count used by the supervised Uvicorn command. The image also sets `WORKERS=1`. |
| `APP_NAME` | `Jev Proxy` | Application name. |
| `ENVIRONMENT` | `development` | Environment label returned by health diagnostics. |
| `PRODUCTION` | `false` | Disables the API documentation routes when `true`. |
| `DEBUG` | `true` | Debug-mode setting. |
| `LOG_LEVEL` | `INFO` | Logging level. |
| `LOG_FORMAT` | `%(asctime)s - %(name)s - %(levelname)s - %(message)s` | Python logging format. |
| `JEV_CATEGORY_PASS` | `true` | Run one Jev Choice per detected span for category classification. |
| `JEV_CATEGORY_CONTEXT` | `2` | Number of before/after context segments used by that pass. |
| `JEV_DEFAULT_CATEGORY` | `sponsor` | Category when the category pass is disabled. |
| `TESTING` | `false` | Testing-mode setting. |

`JEVPROXY_BIND_ADDRESS` is a Compose interpolation variable, not a proxy process setting. It
controls the host side of the `8080:8080` port mapping and defaults to `127.0.0.1`.

---

[< Documentation](README.md) | [Project README](../README.md)
