# Security and storage

[< Documentation index](README.md) | [Project README](../README.md)

## Contents

- [Authentication and secrets](#authentication-and-secrets)
- [Request safety and cache](#request-safety-and-cache)
- [Logging](#logging)
- [Settings storage](#settings-storage)

## Authentication and secrets

A caller bearer token is required by default for inference requests and is forwarded as the TypeSafe key. Set `JEV_ALLOW_UNAUTHENTICATED_FALLBACK=true` only for a protected internal deployment that must permit a configured `TYPESAFE_API_KEY` without a caller bearer token.

`MINUSPOD_PASSWORD` is used for sponsor-list login and for locally authenticating runtime-settings saves. The settings password is not saved in browser storage. See [Configuration](configuration.md) and [Environment variables](environment-variables.md).

## Request safety and cache

- Requests are bounded by `JEV_MAX_CONCURRENT_REQUESTS=4` per worker. The Docker default of 1 worker permits up to 4 concurrent requests. The shim does not apply text, segment, or request-body limits; Jev enforces its current upstream model limits.
- `JEV_REQUEST_DEADLINE_SECONDS=75` is a cooperative request budget covering retries and retry waits. Keep it below nginx's 90 second response-inactivity timeout when changing either value.
- `JEV_CACHE_PATH` is retained as the legacy JSON import source. Active responses are stored in a SQLite sidecar beside it and capped at `JEV_CACHE_MAX_ENTRIES=10000` entries. Failed sponsor refreshes pause for `SPONSOR_FAILURE_COOLDOWN_SECONDS=900` seconds before another attempt. This cooldown is per worker, so account for workers and replicas against MinusPod's login limit.

## Logging

Logs go to stdout at `LOG_LEVEL` (`DEBUG` for verbose tracing). The TypeSafe key, MinusPod password, session cookies, and Authorization header are never logged. Review validation failures log a fixed rule and stage with numeric expected/actual option counts and probability totals, without upstream payloads.

## Settings storage

Saved overrides use `JEV_SETTINGS_PATH` (default `./data/runtime-settings.json`) and survive restarts when that directory is persistent. The Compose file mounts `/app/data` to a named volume. Corrupt or unreadable state returns 503 rather than silently resetting to defaults. Existing deployments must add an equivalent persistent mount; replacing only the image does not preserve the settings file.

The `/app/data` mount must be writable by UID/GID 1000. Existing volume ownership overrides image defaults, so verify the mounted directory before applying a nonrecursive `chown` to UID/GID 1000.
