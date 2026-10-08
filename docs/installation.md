# Installation

[< Documentation index](README.md) | [Project README](../README.md)

## Contents

- [Docker Compose](#docker-compose)
- [Podman Quadlet](#podman-quadlet)
- [Local development](#local-development)
- [Ports and outbound access](#ports-and-outbound-access)

## Docker Compose

The included Compose file starts `ttlequals0/minuspodjev:latest`, mounts the `jevproxy-data` named volume at `/app/data`, and publishes `127.0.0.1:8080` by default.

```bash
docker compose up -d
```

`MINUSPOD_BASE_URL` and `MINUSPOD_PASSWORD` let the proxy name sponsors from MinusPod's live sponsor list. The TypeSafe key rides in the per-request OpenAI bearer token, so it is not normally set in Compose. If you configure `TYPESAFE_API_KEY` as a fallback, callers still need a bearer token unless `JEV_ALLOW_UNAUTHENTICATED_FALLBACK=true`.

For intentional LAN or reverse-proxy exposure, set `JEVPROXY_BIND_ADDRESS=0.0.0.0` in your environment and provide appropriate network access controls.

## Podman Quadlet

A rootless Podman unit that matches the Compose defaults. Save it as `~/.config/containers/systemd/minuspodjev.container`, then run `systemctl --user daemon-reload` and `systemctl --user start minuspodjev`.

```ini
[Unit]
Description=MinusPodJev proxy
After=podman-user-wait-network-online.service
Wants=podman-user-wait-network-online.service

[Container]
Image=docker.io/ttlequals0/minuspodjev:latest
ContainerName=minuspodjev
Network=minuspod.network
PublishPort=127.0.0.1:8080:8080
Volume=%h/jevproxy-data:/app/data:Z
Environment=MINUSPOD_BASE_URL=http://minuspod:8000
Secret=minuspod_password,type=env,target=MINUSPOD_PASSWORD
Environment=JEV_CACHE_PATH=/app/data/jev_cache.json
Environment=JEV_SETTINGS_PATH=/app/data/runtime-settings.json
DropCapability=ALL
AddCapability=CAP_SETUID CAP_SETGID CAP_CHOWN CAP_DAC_OVERRIDE
NoNewPrivileges=true
AutoUpdate=registry

[Service]
Restart=on-failure

[Install]
WantedBy=default.target
```

- `minuspod.network` is the Podman network the MinusPod container is on. `MINUSPOD_BASE_URL` uses that container's name and internal port, and MinusPod's `OPENAI_BASE_URL` becomes `http://minuspodjev:8080/v1`.
- Create the data directory and hand it to container UID 1000, the user uvicorn runs as: `mkdir -p ~/jevproxy-data && podman unshare chown 1000:1000 ~/jevproxy-data`. Drop the `:Z` suffix on hosts without SELinux.
- Create the secret from a file holding the MinusPod login password: `podman secret create minuspod_password /path/to/password-file`. Do not reuse the `MINUSPOD_MASTER_PASSPHRASE` secret here.
- The four capabilities are the tested minimum for the published image. supervisord needs `CAP_SETUID` and `CAP_SETGID` to drop to the app and nginx users; the nginx master process needs `CAP_CHOWN` and `CAP_DAC_OVERRIDE` to create its temp directories under `/var/lib/nginx`. With fewer, nginx exits at startup and the port never opens.
- `ReadOnly=yes` works when these paths are tmpfs: `/run`, `/var/cache/nginx`, `/var/log/nginx`, `/var/lib/nginx/tmp`, and `/tmp`. For example `Mount=type=tmpfs,destination=/run,tmpfs-size=16M` for each one. Images before 0.1.31 need `tmpfs-mode=0755` on the `/run` mount; later images accept the default mode.
- Check with `curl http://127.0.0.1:8080/api/status` and `curl http://127.0.0.1:8080/v1/models`; the second must list `typesafe/jev`.

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
