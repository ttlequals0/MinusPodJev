"""Review-prompt parsing against committed MinusPod corpus transcripts.

These cases build only the review framing around transcript lines captured in
benchmark/data/corpus. They do not call Jev or invent model responses.
"""

import json
import re
from pathlib import Path

from app.services.openai_adapter import (
    parse_candidate_bounds,
    parse_review_context,
    parse_review_segments,
)

CORPUS = Path(__file__).parents[2] / "benchmark" / "data" / "corpus"


def _seconds(value: str) -> float:
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 1:
        return parts[0]
    return parts[-1] + 60 * parts[-2] + (3600 * parts[-3] if len(parts) > 2 else 0)


def _truth_ranges(episode: str) -> list[tuple[float, float]]:
    ranges: list[tuple[float, float]] = []
    start: float | None = None
    for line in (CORPUS / episode / "truth.txt").read_text().splitlines():
        if line.lstrip().startswith("#"):
            continue
        start_match = re.match(r"start:\s*(\S+)", line, re.IGNORECASE)
        if start_match:
            start = _seconds(start_match.group(1))
            continue
        end_match = re.match(r"end:\s*(\S+)", line, re.IGNORECASE)
        if end_match and start is not None:
            ranges.append((start, _seconds(end_match.group(1))))
            start = None
    return ranges


def _segments(episode: str) -> list[dict]:
    return json.loads((CORPUS / episode / "segments.json").read_text())


def _rows_for_range(segments: list[dict], start: float, end: float) -> list[dict]:
    return [segment for segment in segments if segment["start"] < end and segment["end"] > start]


def _render(rows: list[dict]) -> str:
    return "\n".join(f"[{row['start']:.2f}s - {row['end']:.2f}s] {row['text']}" for row in rows)


def _word_lines(rows: list[dict]) -> str:
    return "\n".join(
        f"[{word['start']:.2f}s-{word['end']:.2f}s] {word['word']}"
        for row in rows
        for word in row["words"]
    )


def _review_prompt(
    start: float,
    end: float,
    before: list[dict],
    candidate: list[dict],
    after: list[dict],
    *,
    start_words: list[dict] | None = None,
    end_words: list[dict] | None = None,
) -> str:
    word_timing = ""
    if start_words is not None and end_words is not None:
        word_timing = (
            "Boundary word timing, use these timestamps for corrections:\n"
            "Start edge:\n"
            f"{_word_lines(start_words)}\n"
            "End edge:\n"
            f"{_word_lines(end_words)}\n"
        )
    return (
        "This is the candidate ad to review.\n"
        f"Original boundaries: {start:.2f}s - {end:.2f}s.\n"
        "Transcript (real corpus lines; context may overlap the candidate):\n"
        f"{_render(before)}\n"
        f">>> CANDIDATE AD START [{start:.1f}s] >>>\n"
        f"{_render(candidate)}\n"
        f"<<< CANDIDATE AD END [{end:.1f}s] <<<\n"
        f"{_render(after)}\n"
        f"{word_timing}"
    )


def test_real_adjacent_ads_dedupe_overlapping_context_lines():
    episode = "ep-daily-tech-news-show-c1904b8605f7"
    truth = _truth_ranges(episode)
    assert truth[:2] == [(0.0, 52.32), (52.49, 152.88)]
    segments = _segments(episode)
    candidate = _rows_for_range(segments, *truth[0])
    adjacent = _rows_for_range(segments, *truth[1])
    prompt = _review_prompt(
        *truth[0],
        before=candidate[-1:],
        candidate=candidate,
        after=adjacent,
    )

    parsed = parse_review_segments(prompt)
    keys = [(row["start"], row["end"], row["text"]) for row in parsed]
    assert len(keys) == len(set(keys))
    assert [row["sid"] for row in parsed] == list(range(len(parsed)))
    assert any("Capital One" in row["text"] for row in parsed)
    assert any("Zero" in row["text"] for row in parsed)


def test_real_original_bounds_preserve_subsecond_value_over_rounded_markers():
    episode = "ep-daily-tech-news-show-c1904b8605f7"
    start, end = _truth_ranges(episode)[1]
    segments = _segments(episode)
    prompt = _review_prompt(
        start,
        end,
        before=[],
        candidate=_rows_for_range(segments, start, end),
        after=[],
    )
    assert "Original boundaries: 52.49s - 152.88s." in prompt
    assert ">>> CANDIDATE AD START [52.5s] >>>" in prompt
    assert parse_candidate_bounds(prompt) == (start, end)


def test_real_non_segment_aligned_truth_keeps_word_timed_segment_edges():
    episode = "ep-tosh-show-5f6894439bb6"
    start, end = _truth_ranges(episode)[0]
    segments = _segments(episode)
    candidate = _rows_for_range(segments, start, end)
    containing = [row for row in candidate if row["start"] < end < row["end"]]
    assert containing
    assert containing[0]["words"]
    assert containing[0]["start"] < end < containing[0]["end"]
    assert any(word["end"] == end for word in containing[0]["words"])

    prompt = _review_prompt(
        start,
        end,
        before=[],
        candidate=candidate,
        after=[],
        start_words=candidate[:1],
        end_words=[containing[0]],
    )
    parsed = parse_review_segments(prompt)
    coarse_context, word_context = parse_review_context(prompt)
    assert parse_candidate_bounds(prompt) == (start, end)
    coarse = {(row["start"], row["end"], row["text"]) for row in candidate}
    assert {(row["start"], row["end"], row["text"]) for row in parsed} == coarse
    assert coarse_context == parsed
    expected_start_words = [
        {"start": word["start"], "end": word["end"], "text": word["word"].strip()}
        for word in candidate[0]["words"]
    ]
    expected_end_words = [
        {"start": word["start"], "end": word["end"], "text": word["word"].strip()}
        for word in containing[0]["words"]
    ]
    assert word_context["start"] == expected_start_words
    assert word_context["end"] == expected_end_words
    assert parsed[-1]["end"] == containing[0]["end"]
    assert parsed[-1]["end"] > end


def test_real_no_ad_controls_have_no_truth_candidates():
    for episode in ("ep-ai-cloud-essentials-e8dc897fbd6b", "ep-oxide-and-friends-ce789ff5b62e"):
        assert _truth_ranges(episode) == []
        segments = _segments(episode)
        assert segments
        assert all(segment["words"] for segment in segments)
        prompt = _review_prompt(
            segments[0]["start"],
            segments[0]["end"],
            before=[],
            candidate=segments[:1],
            after=segments[1:2],
        )
        assert len(parse_review_segments(prompt)) == 2


def test_real_zero_width_word_timing_is_not_treated_as_inverted():
    segments = _segments("ep-oxide-and-friends-ce789ff5b62e")
    word = next(word for word in segments[1]["words"] if word["start"] == word["end"])
    prompt = (
        "Boundary word timing, use these timestamps for corrections:\n"
        "Start edge:\n"
        f"[{word['start']:.2f}s-{word['end']:.2f}s] {word['word']}\n"
        "End edge:\n"
    )
    _, words = parse_review_context(prompt)
    assert words["start"] == [{"start": word["start"], "end": word["end"], "text": word["word"].strip()}]
