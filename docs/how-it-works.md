# How it works

[< Documentation index](README.md) | [Project README](../README.md)

## Contents

- [Request flow](#request-flow)
- [Supported MinusPod phases](#supported-minuspod-phases)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)

## Request flow

Jev scores one segment at a time ("is this line an ad?"); the client assembles the spans. The proxy wraps that as an OpenAI chat endpoint:

1. Parse MinusPod's window prompt into segments.
2. Ask Jev one noul per segment.
3. Assemble spans and run a second pass for category.
4. Name sponsors from MinusPod's sponsor list, with a gazetteer fallback.
5. Return the `{"ads": [...]}` JSON in a chat-completion envelope.

## Supported MinusPod phases

One `/chat/completions` handler covers three of MinusPod's four LLM phases:

- **detection**: the window prompt.
- **verification**: re-detection on the re-cut audio, same prompt, handled identically.
- **review**: per-ad prompts (marked `>>> CANDIDATE AD START` / `<<< CANDIDATE AD END`) get Jev's verdict schema (is_ad, boundaries, confidence), including resurrection candidates.

Jev Proxy does not support chapter generation. Configure a secondary chat-model provider and set `chapters_provider` to `secondary`; do not route chapter generation to this proxy.

## Architecture

```mermaid
flowchart TD
  subgraph ops["Operations UI"]
    statuspage["Status Page<br/>[App.tsx]"]
  end

  subgraph proxyapi["Proxy API"]
    health["Health API<br/>[health.py]"]
    status["Status API<br/>[status.py]"]
    stats["Stats API<br/>[status.py]"]
    nativejev["Native Jev API<br/>[jev.py]"]
    openai["OpenAI API<br/>[openai.py]"]
  end

  subgraph pipeline["Detection Pipeline"]
    adapter["Chat Adapter<br/>[openai_adapter.py]"]
    category["Category Pass<br/>[jev.py]"]
    spans["Span Assembly<br/>[spans.py]"]
    jevclient["Jev Client<br/>[jev.py]"]
  end

  subgraph enrich["Enrichment State"]
    cache[("Response Cache<br/>[cache.py]")]
    sponsors["Sponsor Matching<br/>[sponsors.py]"]
    gazetteer["Sponsor Gazetteer<br/>[sponsors.py]"]
  end

  typesafe["TypeSafe Jev"]
  minuspod((MinusPod))
  chatmodel["Chat Model"]
  mpapi["MinusPod API"]

  statuspage -->|loads and Refresh| health
  statuspage -->|loads and Refresh| status
  statuspage -->|polls every 5 s| stats
  status -->|probes Jev| typesafe
  status -->|probes MinusPod| mpapi

  minuspod -->|sends prompts| openai
  openai -->|returns envelope| minuspod
  openai -->|dispatches request| adapter
  adapter -->|returns ad JSON| openai
  nativejev -->|asks segments| jevclient

  adapter -->|classifies spans| category
  adapter -->|assembles spans| spans
  adapter -->|queries Jev| jevclient
  category -->|queries categories| jevclient
  adapter -->|matches sponsors| sponsors

  jevclient -->|sends nouls| typesafe
  typesafe -->|returns probabilities| jevclient
  jevclient -->|caches answers| cache

  sponsors -->|uses gazetteer| gazetteer
  sponsors -->|reads sponsors| mpapi

  minuspod -.->|routes review| chatmodel
  minuspod -.->|routes chapters| chatmodel

  classDef proxyc fill:#dbeafe,stroke:#3b82f6,color:#1e3a8a
  classDef pipec fill:#fde9c8,stroke:#d97706,color:#7c2d12
  classDef enrichc fill:#d1fae5,stroke:#10b981,color:#065f46
  classDef statusc fill:#fee2e2,stroke:#ef4444,color:#991b1b
  class health,status,stats,nativejev,openai proxyc
  class adapter,category,spans,jevclient pipec
  class cache,sponsors,gazetteer enrichc
  class statuspage statusc
```

## Repository layout

- `backend/app/`: the proxy, with `api/` routers and `services/` for Jev calls, detection, review, sponsors, and `openai_adapter.py` for prompt parsing and the ads/verdict envelope.
- `compat/minuspod_compat/`: vendored MinusPod code shared by the proxy and benchmark.
- `benchmark/`: the evaluation harness, corpus, caches, and reports.
- `frontend/`, `deployment/`, `docker-compose.yml`: status page and container stack.
- `JEV_BENCHMARK_REPORT.md`: the test report.
