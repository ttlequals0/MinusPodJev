# Installation

[< Documentation index](README.md) | [Project README](../README.md)

## Contents

- [Docker Compose](#docker-compose)
- [Local development](#local-development)
- [Ports and outbound access](#ports-and-outbound-access)

## Docker Compose

The included Compose file starts `ttlequals0/minuspodjev:latest`, mounts the `jevproxy-data` named volume at `/app/data`, and publishes `127.0.0.1:8080` by default.

```bash
docker compose up -d
```

`MINUSPOD_BASE_URL` and `MINUSPOD_PASSWORD` let the proxy name sponsors from MinusPod's live sponsor list. The TypeSafe key rides in the per-request OpenAI bearer token, so it is not normally set in Compose. If you configure `TYPESAFE_API_KEY` as a fallback, callers still need a bearer token unless `JEV_ALLOW_UNAUTHENTICATED_FALLBACK=true`.

For intentional LAN or reverse-proxy exposure, set `JEVPROXY_BIND_ADDRESS=0.0.0.0` in your environment and provide appropriate network access controls.

## Local development

Requires `uv` and Python 3.11+.

```bash
cp .env.example .env    # TYPESAFE_API_KEY optional; MinusPod sends it per request
uv sync
uv run uvicorn app.main:app --app-dir backend --reload
uv run pytest backend/tests -q   # upstream calls mocked; no network
```

## Ports and outbound access

- The app listens on port 8000. In the container nginx listens on port 8080 and proxies `/api`, `/v1`, `/chat/completions`, and `/models` to it.
- Point MinusPod at `http://<proxy>:8080/v1`, or `http://localhost:8000/v1` when running uvicorn directly.
- Outbound access is HTTPS to `api.typesafe.ai` for Jev and `MINUSPOD_BASE_URL` for sponsors.
