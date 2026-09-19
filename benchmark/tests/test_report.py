"""Report renders cleanly from synthesized calls.jsonl."""
from __future__ import annotations

from benchmark import report
from benchmark.report import charts
from benchmark.report.aggregate import ModelStats
from benchmark.report.sections import _error_bucket, _render_fp_windows
from benchmark.storage import append_jsonl


CALL_RECORD_TEMPLATE = {
    "schema_version": 1,
    "model": "m1",
    "provider_config": "openrouter",
    "underlying_provider": "OpenRouter",
    "episode_id": "ep-001",
    "trial": 0,
    "window_index": 0,
    "temperature": 0.0,
    "prompt_hash": "sha256:abc",
    "response_time_ms": 1500,
    "input_tokens": 1000,
    "output_tokens": 100,
    "total_cost_usd_at_runtime": 0.005,
    "json_format_used": "native",
    "extraction_method": "json_array_direct",
    "compliance_score": 1.0,
    "schema_violations": {"missing_required": 0, "wrong_type": 0, "extra_keys": 0, "out_of_range": 0, "extra_key_names": []},
    "windows_stale": False,
    "error": None,
}


def test_render_with_no_data(tmp_path, minimal_cfg, make_episode, pricing_snapshot):
    calls = tmp_path / "calls.jsonl"
    out = tmp_path / "report.md"
    report.render(
        cfg=minimal_cfg, episodes=[make_episode()],
        calls_path=calls, episode_results_path=tmp_path / "ep.jsonl",
        pricing_snapshot=pricing_snapshot,
        output_path=out, assets_dir=tmp_path / "assets",
    )
    assert "No benchmark data yet" in out.read_text()


def test_render_with_one_call(tmp_path, minimal_cfg, make_episode, pricing_snapshot):
    ep = make_episode(n_windows=1)
    calls = tmp_path / "calls.jsonl"
    append_jsonl(calls, {**CALL_RECORD_TEMPLATE, "call_id": "c1", "parsed_ads": [{"start_time": 0.0, "end_time": 30.0}]})
    out = tmp_path / "report.md"
    report.render(
        cfg=minimal_cfg, episodes=[ep],
        calls_path=calls, episode_results_path=tmp_path / "ep.jsonl",
        pricing_snapshot=pricing_snapshot,
        output_path=out, assets_dir=tmp_path / "assets",
    )
    text = out.read_text()
    assert "## TL;DR" in text
    assert "m1" in text
    assert "Per-Episode Detail" in text
    assert "Run Metadata" in text


def test_per_model_detail_reports_verbosity_and_truncation(tmp_path, minimal_cfg, make_episode, pricing_snapshot):
    """Per-model detail surfaces the over-1024, truncated, and salvaged counts
    so verbose/instruction-resistant models (phi-4, Gemini variants) are
    visible at a glance.
    """
    ep = make_episode(n_windows=1)
    calls = tmp_path / "calls.jsonl"
    append_jsonl(calls, {
        **CALL_RECORD_TEMPLATE, "call_id": "c1", "model": "verbose-model",
        "parsed_ads": [{"start_time": 0.0, "end_time": 30.0}],
        "output_tokens": 1500, "truncated": False, "over_1024_tokens": True,
    })
    append_jsonl(calls, {
        **CALL_RECORD_TEMPLATE, "call_id": "c2", "model": "verbose-model",
        "trial": 1,
        "parsed_ads": [{"start_time": 0.0, "end_time": 30.0}],
        "output_tokens": 4096, "stop_reason": "max_tokens",
        "truncated": True, "over_1024_tokens": True,
        "extraction_method": "json_object_single_ad_truncated",
    })
    out = tmp_path / "report.md"
    report.render(
        cfg=minimal_cfg, episodes=[ep],
        calls_path=calls, episode_results_path=tmp_path / "ep.jsonl",
        pricing_snapshot=pricing_snapshot,
        output_path=out, assets_dir=tmp_path / "assets",
    )
    text = out.read_text()
    assert "over 1024 output tokens" in text
    assert "hit max_tokens" in text
    assert "salvaged from truncated JSON" in text
    # Both calls exceeded 1024 -> 2/2 (100.0%)
    assert "2/2 calls over 1024 output tokens (100.0%)" in text
    # One truncated -> "1 hit max_tokens (50.0%)"
    assert "1 hit max_tokens (50.0%)" in text


def test_json_format_summary_classifies_native_prompt_inject_mixed():
    assert report._json_format_summary({}) == ("n/a", 0.0)
    assert report._json_format_summary({"native": 100}) == ("native", 1.0)
    # 96% native crosses the 95% threshold.
    prim, pct = report._json_format_summary({"native": 96, "prompt_injection": 4})
    assert prim == "native"
    assert pct == 0.96
    # 94% native falls short -> mixed.
    prim, _ = report._json_format_summary({"native": 94, "prompt_injection": 6})
    assert prim == "mixed"
    prim, pct = report._json_format_summary({"prompt_injection": 99, "native": 1})
    assert prim == "prompt-inject"
    assert pct == 0.01


def test_tldr_table_columns_and_json_mode_telemetry(tmp_path, minimal_cfg, make_episode, pricing_snapshot):
    """Best Accuracy table renders the F0.5 tier columns; per-model detail shows JSON mode."""
    ep = make_episode(n_windows=1)
    calls = tmp_path / "calls.jsonl"
    # Two trials on `m1` (one native, one prompt_injection) -> mixed.
    append_jsonl(calls, {
        **CALL_RECORD_TEMPLATE, "call_id": "c1", "trial": 0,
        "parsed_ads": [{"start_time": 0.0, "end_time": 30.0}],
    })
    append_jsonl(calls, {
        **CALL_RECORD_TEMPLATE, "call_id": "c2", "trial": 1,
        "json_format_used": "prompt_injection",
        "parsed_ads": [{"start_time": 0.0, "end_time": 30.0}],
    })
    out = tmp_path / "report.md"
    report.render(
        cfg=minimal_cfg, episodes=[ep],
        calls_path=calls, episode_results_path=tmp_path / "ep.jsonl",
        pricing_snapshot=pricing_snapshot,
        output_path=out, assets_dir=tmp_path / "assets",
    )
    text = out.read_text()
    # Best Accuracy table now leads with F0.5 + paired-tier columns.
    assert "### Best Accuracy (F0.5 @ IoU >= 0.5)" in text
    assert "| Tier | Model | F0.5 @0.5 | F0.5 @0.8 | 95% CI | Precision | Recall | F1 |" in text
    # Per-model detail block surfaces the native percent + call count.
    assert "JSON mode: mixed" in text
    assert "50% native" in text
    assert "2 calls" in text


def test_aggregate_model_order_deterministic(make_episode, pricing_snapshot):
    """Report tables must not inherit set iteration order: with hash
    randomization the same data would render rows in a different order on
    every run, making committed reports un-diffable."""
    ep = make_episode(n_windows=1)
    models = [f"model-{c}" for c in "zyxwvutsrqponmlkjihgfedcba"]
    calls = [
        {**CALL_RECORD_TEMPLATE, "call_id": f"c-{m}", "model": m, "parsed_ads": []}
        for m in models
    ]
    by_model, _ = report._aggregate(calls, [ep], pricing_snapshot=pricing_snapshot)
    assert list(by_model) == sorted(by_model)


def test_render_handles_no_ad_episode(tmp_path, minimal_cfg, make_episode, pricing_snapshot):
    ep = make_episode(n_windows=1, no_ad=True)
    calls = tmp_path / "calls.jsonl"
    append_jsonl(calls, {**CALL_RECORD_TEMPLATE, "call_id": "c2", "parsed_ads": []})
    out = tmp_path / "report.md"
    report.render(
        cfg=minimal_cfg, episodes=[ep],
        calls_path=calls, episode_results_path=tmp_path / "ep.jsonl",
        pricing_snapshot=pricing_snapshot,
        output_path=out, assets_dir=tmp_path / "assets",
    )
    text = out.read_text()
    assert "PASS" in text
    assert "no-ads" in text.lower() or "no-ad" in text.lower()


def test_chart_svg_output_is_deterministic(tmp_path):
    """Committed report assets must be byte-stable when data is unchanged:
    matplotlib element ids and embedded dates would otherwise churn every
    regen and destroy the audit diff."""
    s = ModelStats(model="m1")
    s.avg_f1 = 0.5
    s.total_episode_cost = 0.01
    charts._render_pareto({"m1": s}, tmp_path / "a.svg")
    charts._render_pareto({"m1": s}, tmp_path / "b.svg")
    assert (tmp_path / "a.svg").read_bytes() == (tmp_path / "b.svg").read_bytes()


def test_moderation_block_is_counted_and_flagged():
    """A content-moderation refusal must be counted before errored calls are
    skipped, and must flag the row: the score above it excludes those windows."""
    from benchmark.report.aggregate import _is_moderation_block
    from benchmark.report.sections import _moderation_pct, _reliability_flags
    from benchmark.report.aggregate import ModelStats

    blocked = {"message": "Error code: 451 - {'error': {'message': "
                          "'The content you provided or machine outputted is blocked.', "
                          "'type': 'censorship_blocked'}}"}
    assert _is_moderation_block(blocked) is True
    assert _is_moderation_block({"message": "Insufficient credits."}) is False
    assert _is_moderation_block({"message": "Rate limit exceeded."}) is False
    assert _is_moderation_block({"message": "Expecting value: line 1 column 1"}) is False

    s = ModelStats(model="m", moderation_blocked=130, attempted_count=855)
    assert abs(_moderation_pct(s) - 0.15204) < 1e-4
    assert "moderation blocked 15.2%" in _reliability_flags(s)

    clean = ModelStats(model="m", moderation_blocked=0, attempted_count=855)
    assert _moderation_pct(clean) == 0.0
    assert "moderation" not in _reliability_flags(clean)


def test_campaign_mixing_detects_two_prompt_hashes_per_unit():
    """calls.jsonl accumulates campaigns unless rotated. Dedup ignores
    prompt_hash, so a partial re-run silently keeps old rows for the units it
    did not reach. Nothing else in the report would surface that."""
    from benchmark.report.aggregate import campaign_mixing

    def row(model, w, h):
        return {"model": model, "episode_id": "ep", "trial": 0,
                "window_index": w, "prompt_hash": h}

    # one unit run under two prompts, one unit run twice under the same prompt
    calls = [row("m", 0, "sha256:old"), row("m", 0, "sha256:new"),
             row("m", 1, "sha256:new"), row("m", 1, "sha256:new")]
    assert campaign_mixing(calls) == {"m": 1}

    # a clean single-campaign file reports nothing
    assert campaign_mixing([row("m", 0, "sha256:new"), row("m", 1, "sha256:new")]) == {}


def test_error_bucket_classifies_moderation_and_account_errors():
    """The failure table's classifier must agree with the aggregate-side
    moderation detector; a 451 censorship block landing in Other hides the
    single most production-relevant failure mode."""
    assert _error_bucket(
        "Error code: 451 - {'type': 'censorship_blocked', 'message': "
        "'The content you provided or machine outputted is blocked.'}"
    ) == "Provider content moderation rejection"
    assert _error_bucket("Error code: 402 - This request requires more credits") == "Credits exhausted"
    assert _error_bucket("Error code: 403 - complete the following before use: 18+ age confirmation") == "Account gating (age confirmation)"
    assert _error_bucket("Error code: 429 - rate limit exceeded") == "Rate-limited"
    assert _error_bucket("Expecting value: line 1 column 1") == "Other"


def test_provider_policy_block_is_not_an_unknown_model():
    """OpenRouter returns 404 when the operator's account blocks a provider.
    Bucketing that as a bad slug sends readers to check a correct model id."""
    assert _error_bucket(
        "Error code: 404 - No endpoints available matching your guardrail restrictions and data policy"
    ) == "Account gating (provider policy)"
    assert _error_bucket(
        "Error code: 404 - No allowed providers are available for this model"
    ) == "Account gating (provider policy)"
    assert _error_bucket("Error code: 404 - No such model") == "Unknown model (404)"
    # A model with no provider at all is unavailable, not gated by the account.
    assert _error_bucket(
        "Error code: 404 - No endpoints found for acme/some-model"
    ) == "Unknown model (404)"


def test_fp_windows_table_lists_only_truthless_windows(make_episode):
    """A window overlapping a truth ad must not appear no matter how many
    models flagged it; a truthless window needs 2+ votes to make the table."""
    ep = make_episode("ep-a", n_windows=2)          # truth ad at 0-10s, w0 spans 0-300
    control = make_episode("ep-b", n_windows=1, no_ad=True)
    agreement = {
        ("ep-a", 0): {"m1": 3, "m2": 1},            # overlaps the truth ad
        ("ep-a", 1): {"m1": 2, "m2": 1, "m3": 0},   # truthless, 2 of 3 voted
        ("ep-b", 0): {"m1": 1, "m2": 1},            # no-ad control, tagged
    }
    text = "\n".join(_render_fp_windows(agreement, 3, [ep, control]))
    assert "### Windows flagged with no truth ad" in text
    assert "| `ep-a` | 1 | 300-600s | 2 of 3 |" in text
    assert "| `ep-a` | 0 " not in text
    assert "| `ep-b` (no-ad control) | 0 | 0-600s | 2 of 3 |" in text

    solo_vote = {("ep-a", 1): {"m1": 1, "m2": 0, "m3": 0}}
    assert _render_fp_windows(solo_vote, 3, [ep]) == []


def test_cost_columns_are_per_episode_not_corpus_totals(make_episode, pricing_snapshot):
    """`Cost / episode` is documented as an average per episode. Summing the
    per-episode means across the corpus put the chat rows on a different
    denominator than the jev row in the same column."""
    eps = [make_episode("ep-001", n_windows=1), make_episode("ep-002", n_windows=1)]
    calls = [
        {**CALL_RECORD_TEMPLATE, "call_id": f"c-{e.ep_id}", "episode_id": e.ep_id,
         "parsed_ads": [{"start_time": 0.0, "end_time": 30.0}]}
        for e in eps
    ]
    by_model, _ = report._aggregate(calls, eps, pricing_snapshot=pricing_snapshot)
    s = by_model["m1"]
    # 1000 in @ $3/Mtok + 100 out @ $15/Mtok = $0.0045 per episode, on both episodes.
    assert s.cost_episodes == 2
    assert abs(s.total_episode_cost - 0.0045) < 1e-9
    assert abs(s.input_episode_cost - 0.003) < 1e-9
    assert abs(s.output_episode_cost - 0.0015) < 1e-9
    # The TL;DR total and the cost-breakdown split must be the same figure.
    assert abs(s.input_episode_cost + s.output_episode_cost - s.total_episode_cost) < 1e-9
    # Cost / TP stays a corpus ratio: per-episode cost re-multiplied by the episode count.
    s.tp_total = 3
    assert abs(s.cost_per_tp - 0.009 / 3) < 1e-9


def test_jev_cost_uses_the_same_per_episode_denominator(tmp_path, write_corpus_episode):
    """The jev row and the chat rows must normalize `Cost / episode` the same way."""
    import json

    from benchmark import corpus, jev

    eps = [corpus.load_episode(write_corpus_episode(tmp_path, ep_id=e)) for e in ("ep-a", "ep-b")]
    entries = {}
    for ep in eps:
        for segs in jev.episode_windows(ep):
            key = jev.hash_payload(jev.build_payload(
                segs, guidance=jev.GUIDANCE_FULL, metadata=ep.metadata))
            entries[key] = {
                "probabilities": {f"s{g['sid']}": 0.9 for g in segs},
                "input_tokens": 1000, "output_tokens": 0, "elapsed_ms": 0,
            }
    cache = tmp_path / "jev_cache.json"
    cache.write_text(json.dumps(entries))

    stats = {}
    jev.merge_into_stats(stats, eps, cache_path=cache)
    s = stats["jev"]
    assert s.cost_episodes == len(eps)
    assert abs(s.total_episode_cost - 1000 * jev.INPUT_COST_PER_MTOK / 1e6) < 1e-12
    assert abs(s.input_episode_cost + s.output_episode_cost - s.total_episode_cost) < 1e-12
    # No elapsed_ms in the cache -> no timings, and no trials -> no stdev.
    assert s.p50_call_latency_ms == 0.0
    assert s.f1_stdev_per_episode == {}


def test_latency_tail_marks_untimed_rows_na_and_sorts_them_last():
    """A row with no timing data defaults to 0.0ms. Printed as `0.0s` it reads
    as the fastest system in the benchmark."""
    from benchmark.report.sections import _render_latency_tail

    timed = ModelStats(model="timed")
    timed.p50_call_latency_ms = 1500.0
    timed.p90_call_latency_ms = 2000.0
    timed.p95_call_latency_ms = 2500.0
    timed.p99_call_latency_ms = 3000.0
    timed.max_call_latency_ms = 4000.0
    untimed = ModelStats(model="untimed")

    text = _render_latency_tail({"untimed": untimed, "timed": timed})
    assert "| `untimed` | n/a | n/a | n/a | n/a | n/a |" in text
    assert "| `timed` | 1.50s | 2.00s | 2.50s | 3.00s | 4.00s |" in text
    assert "0.00s" not in text
    assert text.index("`timed`") < text.index("`untimed`")


def test_tldr_latency_column_marks_untimed_rows_na(tmp_path, minimal_cfg, make_episode, pricing_snapshot):
    ep = make_episode(n_windows=1)
    calls = tmp_path / "calls.jsonl"
    append_jsonl(calls, {**CALL_RECORD_TEMPLATE, "call_id": "c1", "response_time_ms": 0,
                         "parsed_ads": [{"start_time": 0.0, "end_time": 30.0}]})
    out = tmp_path / "report.md"
    report.render(
        cfg=minimal_cfg, episodes=[ep],
        calls_path=calls, episode_results_path=tmp_path / "ep.jsonl",
        pricing_snapshot=pricing_snapshot,
        output_path=out, assets_dir=tmp_path / "assets",
    )
    text = out.read_text()
    assert "| 0.0s |" not in text
    assert "p50 / p95 latency: n/a / n/a" in text


def test_single_trial_leaves_f1_stdev_unpopulated(make_episode, pricing_snapshot):
    """Stdev over n=1 is undefined. Storing 0.0 publishes the default as a
    measurement and ranks the row as the most deterministic in the benchmark."""
    ep = make_episode(n_windows=1)
    one = [{**CALL_RECORD_TEMPLATE, "call_id": "c1", "parsed_ads": []}]
    two = one + [{**CALL_RECORD_TEMPLATE, "call_id": "c2", "trial": 1, "parsed_ads": []}]
    single, _ = report._aggregate(one, [ep], pricing_snapshot=pricing_snapshot)
    paired, _ = report._aggregate(two, [ep], pricing_snapshot=pricing_snapshot)
    assert single["m1"].f1_stdev_per_episode == {}
    assert paired["m1"].f1_stdev_per_episode.keys() == {ep.ep_id}


def test_trial_variance_marks_rows_without_two_trials_na():
    from benchmark.report.sections import _render_trial_variance

    measured = ModelStats(model="measured")
    measured.avg_f1 = 0.5
    measured.f1_per_episode = {"ep-001": 0.5}
    measured.f1_stdev_per_episode = {"ep-001": 0.0375}
    single = ModelStats(model="single")
    single.avg_f1 = 0.99
    single.f1_per_episode = {"ep-001": 0.99}

    text = _render_trial_variance({"single": single, "measured": measured})
    assert "| `single` | n/a | n/a |" in text
    assert "| `measured` | 0.0375 | 0.0375 |" in text
    assert "0.0000" not in text
    # The top-scoring row has no stdev; it must not lead the determinism table.
    assert text.index("`measured`") < text.index("`single`")


def test_strict_f05_uses_same_estimator_at_a_higher_threshold(make_episode, pricing_snapshot):
    """A span that clears IoU 0.5 but not 0.8 scores full credit in the loose
    column and zero in the strict one. Truth is 0-10s; 0-14s is IoU 0.71."""
    ep = make_episode(n_windows=1)
    exact = [{**CALL_RECORD_TEMPLATE, "call_id": "c1", "parsed_ads": [{"start_time": 0.0, "end_time": 10.0}]}]
    loose = [{**CALL_RECORD_TEMPLATE, "call_id": "c2", "parsed_ads": [{"start_time": 0.0, "end_time": 14.0}]}]
    tight, _ = report._aggregate(exact, [ep], pricing_snapshot=pricing_snapshot)
    slack, _ = report._aggregate(loose, [ep], pricing_snapshot=pricing_snapshot)
    assert tight["m1"].avg_f05 == 1.0
    assert tight["m1"].avg_f05_strict == 1.0
    assert slack["m1"].avg_f05 == 1.0
    assert slack["m1"].avg_f05_strict == 0.0
    assert slack["m1"].f05_strict_per_episode.keys() == slack["m1"].f05_per_episode.keys()


def test_strict_f05_averages_per_trial_then_per_episode(make_episode, pricing_snapshot):
    """Same two-step mean as the loose column: one trial matches strictly, the
    other does not, so the episode carries 0.5 rather than a pooled count."""
    ep = make_episode(n_windows=1)
    calls = [
        {**CALL_RECORD_TEMPLATE, "call_id": "c1", "trial": 0, "parsed_ads": [{"start_time": 0.0, "end_time": 10.0}]},
        {**CALL_RECORD_TEMPLATE, "call_id": "c2", "trial": 1, "parsed_ads": [{"start_time": 0.0, "end_time": 14.0}]},
    ]
    stats, _ = report._aggregate(calls, [ep], pricing_snapshot=pricing_snapshot)
    assert stats["m1"].avg_f05 == 1.0
    assert stats["m1"].avg_f05_strict == 0.5


def test_tldr_renders_both_f05_thresholds_and_na_without_strict_data():
    from benchmark.report.sections import _render_tldr

    class _Ep:
        class _T:
            is_no_ad_episode = False
        truth = _T()

    def _row(model, strict):
        s = ModelStats(model=model)
        s.avg_f05 = 0.90
        s.f05_per_episode = {"e0": 0.89, "e1": 0.91}
        s.avg_precision = s.avg_recall = s.avg_f1 = 0.9
        s.total_episode_cost = 1.0
        s.json_compliance_mean = 1.0
        s.avg_f05_strict = strict
        return s

    out = _render_tldr({"scored": _row("scored", 0.42), "unscored": _row("unscored", None)}, [_Ep()])
    assert "| Tier | Model | F0.5 @0.5 | F0.5 @0.8 | 95% CI |" in out
    assert "| `scored` | 0.900 | 0.420 |" in out
    assert "| `unscored` | 0.900 | n/a |" in out
