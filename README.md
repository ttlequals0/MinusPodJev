# MinusPodJev

Two things live here:

1. **jevproxy** (`backend/`): a FastAPI service that makes TypeSafe Jev look like an OpenAI
   chat model, so MinusPod can point at it and cut ads with Jev instead of a chat LLM.
2. **benchmark** (`benchmark/`): the offline harness that measured Jev against 84 chat
   models on a 14-episode corpus, plus the caches and the generated reports. It runs
   standalone with `uv`.

The full test writeup is `JEV_BENCHMARK_REPORT.md`. Its short version: Jev ties the best
chat model within noise, costs about 36x less and runs about 59x faster, never cuts audio
that has no ad in it, but cuts a whole segment at a time when it errs. It does not beat chat
models on accuracy once you score at a strict IoU or cross-validate its tuned thresholds.

## Flow

```mermaid
flowchart TD
  op["Episode ingest"] --> mp

  subgraph mp["MinusPod pipeline (per-phase provider routing)"]
    det["detection"]
    ver["verification"]
    rev["review"]
    chap["chapters"]
  end

  det -->|primary| proxy
  ver -->|primary| proxy
  rev -->|primary| proxy
  chap -->|secondary| real["Real chat model"]

  subgraph proxy["jevproxy  POST /v1/chat/completions"]
    disc{"candidate markers or review prompt?"}
    parse["parse Xs-Ys lines to segments"]
    detf["detection: assemble spans + category"]
    revf["review: verdict"]
    spon["sponsor_name: match MinusPod /sponsors, else jev-uuid"]
    disc -->|no| parse --> detf --> spon
    disc -->|yes| revf
  end

  proxy -->|"noul per segment, no model field, bearer = TypeSafe key"| jev[("TypeSafe Jev System One")]
  jev -->|per-segment probabilities| proxy
  spon -.->|"GET /sponsors (MINUSPOD_API_TOKEN)"| mpapi[("MinusPod API")]

  proxy -->|"ads / verdict JSON in chat.completion"| mp
  mp --> cut["cut or replace audio"]
  mp --> learn["pattern learning"]
```

## jevproxy

Jev scores one segment at a time ("is this line an ad?") and the client assembles the spans.
The proxy hides that behind the OpenAI chat-completions API: it parses MinusPod's window
prompt back into segments, asks Jev one noul per segment, assembles spans, runs a second
pass for the ad category, matches sponsors against MinusPod's gazetteer, and returns the
`{"ads": [...]}` JSON MinusPod expects, wrapped in a chat-completion envelope.

The same `/chat/completions` handler covers three of MinusPod's four LLM phases:

- **detection** - the window prompt above.
- **verification** - MinusPod's verification pass re-detects on the re-cut audio with the
  same window prompt, so it is handled identically.
- **review** - per-ad review prompts (marked with `>>> CANDIDATE AD START` / `<<< CANDIDATE
  AD END`) are detected by those markers and answered with Jev's verdict schema (is_ad,
  boundaries, confidence). The resurrection pool is handled the same way; trim-recovery, which
  needs text generation, degrades to a safe no-change verdict.

Chapter generation is the fourth phase and Jev cannot do it (it is generative, not a
per-segment judgment), so chapters route to a real model instead. See the routing table.

Endpoints:
- `POST /v1/chat/completions` and `POST /chat/completions` - the OpenAI-compatible surface
  MinusPod calls (mounted both with and without `/v1`); handles detection, verification, and
  review by inspecting the prompt
- `GET /v1/models`, `GET /models` - advertise `typesafe/jev`
- `POST /api/v1/jev/ask` - the native segments-in, spans-out endpoint
- `GET /api/health`, `GET /api/docs`

### Pointing MinusPod at it (no app-code change)

MinusPod resolves a provider per pipeline phase (`llm_route.py`), so this is all settings.

Set the **primary** provider to the proxy: `openai_compatible`, base URL your proxy (with or
without `/v1`), model `typesafe/jev`, addressing mode `timestamps`, and API key = your
**TypeSafe key**. The proxy forwards that bearer token upstream to Jev, so the key is
configured once in MinusPod (it also honors a `TYPESAFE_API_KEY` env var as a fallback for
direct callers). Set a **secondary** provider to a real chat model (its own base URL and key)
for the one phase Jev cannot do.

| phase | provider slot | goes to |
|---|---|---|
| `detection_provider` | primary | proxy -> Jev |
| `verification_provider` | primary | proxy -> Jev |
| `review_provider` | primary | proxy -> Jev |
| `chapters_provider` | secondary | a real chat model |

The proxy advertises `typesafe/jev` on `/models`, maps it to the upstream Jev model
internally, reads `TYPESAFE_API_KEY` from its own environment, and returns HTTP 503 with a
clear message when it is unset.

Sponsor naming: set `MINUSPOD_SPONSORS_URL` to MinusPod's `GET /sponsors` (and
`MINUSPOD_API_TOKEN` if that call needs auth) so the proxy fills each ad's `sponsor_name`
from the live sponsor list by matching name and aliases. A cut with no match gets a unique
`jev-<7char>` placeholder, which keeps MinusPod's pattern creation from clustering unnamed
ads. With the URL unset it falls back to the vendored gazetteer.

Two review tunables, both defaulted sensibly and left in `config.py`: the review route reuses
`JEV_ENTER`/`JEV_STAY`, and it emits Jev's segment-edge boundaries (so a disagreement reads
as an "adjust", which MinusPod clamps to `max_boundary_shift_seconds`). Flip it to
confirm-in-place if you would rather the review never move a boundary.

### Run it

Requires `uv` and Python 3.11+.

```bash
cp .env.example .env    # set TYPESAFE_API_KEY
uv sync
uv run uvicorn app.main:app --app-dir backend --reload
uv run pytest backend/tests -q   # upstream calls mocked; no network
```

## benchmark

Runs from `benchmark/` with its own `uv` project. It reuses the vendored MinusPod pieces in
`compat/` (window building, the ad schema, the sponsor gazetteer, pricing), so it does not
depend on a MinusPod checkout.

```bash
cd benchmark
uv sync
uv run pytest -q                                   # 298 tests
uv run benchmark jev-spike --oracle off --passes 1 # reproduces F0.5 0.957 from cache
uv run benchmark combined-report --jev-passes 1    # regenerates results/report-combined.md
```

Reports and data:
- `benchmark/results/report-combined.md` - the ranking: Jev and Jev-ceiling against all 84
  chat models, every table, with an `F0.5 @0.8` column beside the headline `F0.5 @0.5`
- `benchmark/results/report.md` - the segment_ids-mode report
- `benchmark/results/audit-2026-09-19.md` - the working investigation log
- `benchmark/results/raw/jev_cache.json` - Jev probabilities, 1026 entries (1-pass + 5-pass,
  live from TypeSafe)
- `benchmark/results/raw/haiku_persegment_cache.json` - Haiku run through Jev's decomposition
  (the ablation), 171 windows

`jev-spike` and `combined-report` read the cache, so they cost nothing to rerun. A fresh
run needs `TYPESAFE_API_KEY`.

## Layout

- `backend/app/api/` - routers: OpenAI-compat, native `jev/ask`, health
- `backend/app/services/` - Jev payload building, detection, category second pass
- `backend/app/services/openai_adapter.py` - prompt parsing, sponsor match, ads/envelope build
- `compat/minuspod_compat/` - vendored MinusPod code the proxy and benchmark share
- `benchmark/` - the evaluation harness, corpus, caches, and reports
- `frontend/`, `deployment/`, `docker-compose*.yml` - proxy status page and container stack
- `JEV_BENCHMARK_REPORT.md` - the test report
- `jev-benchmark-archive-2026-09-19.zip` - a snapshot of the report, results, and analysis tools
