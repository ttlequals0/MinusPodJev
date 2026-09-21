# Benchmark

[< Documentation index](README.md) | [Project README](../README.md)

## Contents

- [Scope and result caveat](#scope-and-result-caveat)
- [Reproduce](#reproduce)
- [Reports and data](#reports-and-data)

## Scope and result caveat

The offline harness measured Jev against 84 chat models on a 14-episode corpus. On that corpus, Jev ties the best chat model within noise at about 36x lower cost and about 59x lower latency, never cuts ad-free audio, but cuts a whole segment at a time when wrong. It does not beat chat models on accuracy at strict IoU or after cross-validating its tuned thresholds.

The full analysis is [JEV_BENCHMARK_REPORT.md](../JEV_BENCHMARK_REPORT.md). The repository report is [benchmark/results/report-combined.md](../benchmark/results/report-combined.md). The generated report and its assets remain under [`benchmark/results/`](../benchmark/results/).

## Reproduce

The benchmark runs from `benchmark/` with its own `uv` project. It reuses the vendored MinusPod pieces in `compat/`, including windows, ad schema, sponsor gazetteer, and pricing, so it needs no MinusPod checkout.

```bash
cd benchmark
uv sync
uv run pytest -q                                   # 298 tests
uv run benchmark jev-spike --oracle off --passes 1 # reproduces F0.5 0.957 from cache
uv run benchmark combined-report --jev-passes 1    # regenerates results/report-combined.md
```

`jev-spike` and `combined-report` read the cache, so reruns cost nothing; a fresh run needs `TYPESAFE_API_KEY`.

## Reports and data

Reports and data under [`benchmark/results/`](../benchmark/results/):

- [`report-combined.md`](../benchmark/results/report-combined.md): Jev and Jev-ceiling ranked against all 84 chat models, with an `F0.5 @0.8` column beside the headline `F0.5 @0.5`.
- [`report.md`](../benchmark/results/report.md): the segment_ids-mode report.
- [`audit-2026-09-19.md`](../benchmark/results/audit-2026-09-19.md): the working investigation log.
- [`raw/jev_cache.json`](../benchmark/results/raw/jev_cache.json): Jev probabilities, 1026 entries (1-pass + 5-pass, live).
- [`raw/haiku_persegment_cache.json`](../benchmark/results/raw/haiku_persegment_cache.json): Haiku through Jev's decomposition (the ablation).
