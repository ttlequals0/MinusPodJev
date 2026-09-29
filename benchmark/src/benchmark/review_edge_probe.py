"""Replay archived reviewer intervals against whole-cut and local edge questions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

from app.services.jev import GUIDANCE, jev_review_questions
from dotenv import dotenv_values

MODELS = ("jev-latest", "jev-1.13.0")
URL = "https://api.typesafe.ai/v1/systemone"
REVIEW_GUIDANCE = GUIDANCE.replace(
    "Each line of `transcript` is one segment of a podcast episode, prefixed with its line id. A line is ADVERTISING when it is",
    "The `target_speech` is from a podcast episode. Speech is ADVERTISING when it is",
).replace("A line is EDITORIAL CONTENT when it is", "Speech is EDITORIAL CONTENT when it is")
if REVIEW_GUIDANCE == GUIDANCE or "line of `transcript`" in REVIEW_GUIDANCE:
    raise RuntimeError("review guidance still refers to detection lines")
AD_ONLY = {
    "type": "noul",
    "instructions": "All speech inside `original_speech` is advertising or promotional content in this break under `guidance`, including sponsor introductions, product anecdotes, offers, and sign-offs. Unrelated show teases, discussion, or returns are not advertising. Judge only speech inside the interval; omitted advertising words affect the comparison, not this safety answer.",
}
EDGE_AD = (
    "All words in `target_speech` are advertising or promotional content under "
    "`guidance`, including sponsor introductions, product anecdotes, offers, "
    "URLs, and sign-offs. Unrelated show discussion, teases, and returns are "
    "not advertising. Judge the target words, using neighboring speech only "
    "to interpret them."
)
EDGE_SHOW = (
    "The words in `target_speech` return to unrelated show content under "
    "`guidance`, rather than continuing the advertising or promotional break. "
    "Judge the target words, using neighboring speech only to interpret them."
)
EDGE_RISK = (
    "`target_speech` includes words that should remain outside an advertising "
    "cut because they return to the show, tease unrelated show content, or "
    "resume editorial discussion under `guidance`. A return-to-show "
    "announcement stays outside the cut even if the advertiser says it. "
    "Ordinary sponsor thanks, offers, URLs, and product-pitch anecdotes remain "
    "part of advertising. Decide only about target words, using neighboring "
    "speech to understand them."
)
EDGE_ROLE = {
    "type": "choice",
    "instructions": (
        "What function does `target_speech` serve in context under `guidance`? "
        "Classify the target words, using neighboring speech only to interpret "
        "them. A return-to-show announcement is show content even when the "
        "advertiser speaks it."
    ),
    "criteria": {
        "ad_break": "The target is part of the advertising break, including its narrative setup, sponsor read, offer, URL, or ordinary sign-off.",
        "show_content": "The target resumes or teases unrelated show content, including an announcement that the show is returning.",
        "unclear": "The target mixes both functions or the neighboring speech cannot settle its role.",
    },
}
UNIT_ROLE = {
    "type": "choice",
    "instructions": (
        "What function does only `target_speech` serve in this podcast under "
        "`guidance`? Use neighboring speech to interpret the target, but do "
        "not classify the neighbors. If the target mixes functions, choose unclear."
    ),
    "criteria": {
        "promotional_content": "The target is a sponsor or promotional pitch, narrative setup, product anecdote, offer, URL, or sponsor thanks.",
        "break_transition": "The target briefly announces the start or return from an ad break without substantive unrelated show discussion.",
        "substantive_show": "The target is unrelated show discussion, a show tease, or other substantive editorial speech.",
        "unclear": "The target mixes these functions or the available speech cannot settle its role.",
    },
}
COREF_UNIT_ROLE = {
    **UNIT_ROLE,
    "criteria": {
        **UNIT_ROLE["criteria"],
        "promotional_content": (
            UNIT_ROLE["criteria"]["promotional_content"]
            + " Resolve pronouns from nearby speech. A brief remark about a sponsor's support or relationship, continuing sponsor thanks, remains a promotional sign-off even if conversational; classify later independent discussion separately."
        ),
    },
}
RELATION_QUESTIONS = {
    "same_promotion": {
        "question": "Does target_speech belong to the same sponsor promotion as nearby_candidate_speech?",
        "true": "The target is the sponsor pitch, its narrative setup, a sponsor-related personal aside, offer, URL, or thanks. Resolve pronouns from the nearby speech; the target need not repeat a brand name.",
        "false": "The target is unrelated episode discussion, a show tease, or only break navigation. Adjacency to a sponsor is insufficient.",
    },
    "substantive_show": {
        "question": "Does target_speech contribute unrelated episode subject matter?",
        "true": "The target discusses or teases the show's subject independently of the sponsor promotion.",
        "false": "The target is sponsor-related content or only a brief announcement that a break is starting or ending.",
    },
    "break_navigation": {
        "question": "Does target_speech only announce that an ad break is starting or ending?",
        "true": "The target is brief break navigation without a sponsor pitch or substantive episode discussion.",
        "false": "The target contains a sponsor message, sponsor thanks, or substantive show discussion.",
    },
}


def selected(words: list[dict], start: float, end: float) -> list[dict]:
    return [word for word in words if float(word["end"]) > start and float(word["start"]) < end]


def speech(words: list[dict]) -> str:
    return " ".join(str(word["word"]).strip() for word in words).strip()


def edge_state(words: list[dict], target: list[dict], context_words: int) -> dict[str, str]:
    first = words.index(target[0])
    last = words.index(target[-1])
    return {
        "guidance": GUIDANCE,
        "before_speech": speech(words[max(0, first - context_words):first]),
        "target_speech": speech(target),
        "after_speech": speech(words[last + 1:last + 1 + context_words]),
    }


def directional_edge_state(words: list[dict], target: list[dict], name: str) -> tuple[dict[str, str], dict[str, int | str]]:
    first = words.index(target[0])
    last = words.index(target[-1])
    before = words[max(0, first - 64):first]
    after = words[last + 1:last + 65]
    if name == "changed_phrase":
        state = {
            "guidance": GUIDANCE,
            "before_speech": speech(before[-32:]),
            "target_speech": speech(target),
            "after_speech": speech(after[:32]),
        }
        return state, {"before_context_words": min(len(before), 32), "after_context_words": min(len(after), 32), "target_words": len(target)}
    inside_is_after = name in {"start_inside", "before_start"}
    inside = after if inside_is_after else before
    outside = before[-16:] if inside_is_after else after[:16]
    state = {
        "guidance": GUIDANCE,
        "inside_context": speech(inside),
        "target_speech": speech(target),
        "outside_context": speech(outside),
    }
    return state, {
        "inside_side": "after" if inside_is_after else "before",
        "inside_context_words": len(inside),
        "outside_context_words": len(outside),
        "target_words": len(target),
    }


def timed_unit(words: list[dict], anchor: int, lower: int, upper: int) -> tuple[list[dict], dict[str, str]]:
    def terminal(word: dict) -> bool:
        return re.search(r"[.!?][\"']?$", str(word["word"]).strip()) is not None

    lo = hi = anchor
    left_stop = "candidate"
    right_stop = "candidate"
    while lo > lower:
        previous = words[lo - 1]
        if previous["_segment"] != words[lo]["_segment"]:
            left_stop = "row"
            break
        if terminal(previous):
            left_stop = "punctuation"
            break
        if float(words[lo]["start"]) - float(previous["end"]) >= 0.6:
            left_stop = "pause"
            break
        lo -= 1
    while hi < upper:
        following = words[hi + 1]
        if terminal(words[hi]):
            right_stop = "punctuation"
            break
        if following["_segment"] != words[hi]["_segment"]:
            right_stop = "row"
            break
        if float(following["start"]) - float(words[hi]["end"]) >= 0.6:
            right_stop = "pause"
            break
        hi += 1
    if terminal(words[hi]):
        right_stop = "punctuation"
    return words[lo:hi + 1], {"left_stop": left_stop, "right_stop": right_stop}


def build_case(case: dict, *, context_words: int = 12, timed_units: bool = False, directional: bool = False, relation: bool = False, review_guidance: bool = False, selected_edges: set[str] | None = None) -> dict:
    source = Path(case["source"]).expanduser().resolve()
    if not source.is_file() or not source.is_relative_to(Path.home() / "Downloads"):
        raise ValueError("source must be an archived file under Downloads")
    raw = source.read_bytes()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    if case.get("source_sha256") != source_sha256:
        raise ValueError("source SHA256 does not match the private manifest")
    document = json.loads(raw)
    words = sorted(
        ({**word, "_segment": index} for index, segment in enumerate(document["segments"]) for word in segment["words"]),
        key=lambda word: (float(word["start"]), float(word["end"])),
    )
    start, end = float(case["start"]), float(case["end"])
    if not 0 <= start < end:
        raise ValueError("invalid candidate bounds")
    inside = selected(words, start, end)
    if not inside:
        raise ValueError("candidate has no timed words")
    before = [word for word in words if float(word["end"]) <= start]
    after = [word for word in words if float(word["start"]) >= end]
    targets = {
        "start_inside": inside[:8],
        "end_inside": inside[-8:],
        "before_start": before[-8:],
        "after_end": after[:8],
    }
    unit_meta = {}
    if timed_units:
        first, last = words.index(inside[0]), words.index(inside[-1])
        for name, anchor, lower, upper in (
            ("start_inside", first, first, last),
            ("end_inside", last, first, last),
            ("before_start", first - 1, 0, first - 1),
            ("after_end", last + 1, last + 1, len(words) - 1),
        ):
            if 0 <= anchor < len(words):
                targets[name], unit_meta[name] = timed_unit(words, anchor, lower, upper)
    if "target_start" in case and "target_end" in case:
        targets["changed_phrase"] = selected(words, float(case["target_start"]), float(case["target_end"]))
        if not targets["changed_phrase"]:
            raise ValueError("changed phrase has no timed words")
    edge = {}
    for name, target in targets.items():
        if not target or (selected_edges is not None and name not in selected_edges):
            continue
        if directional:
            state, context_meta = directional_edge_state(words, target, name)
        else:
            state, context_meta = edge_state(words, target, context_words), {}
        if relation:
            reference = selected(words, start, end)
            target_start, target_end = float(target[0]["start"]), float(target[-1]["end"])
            if target_start >= end:
                reference = reference[-64:]
            elif target_end <= start:
                reference = reference[:64]
            elif name in {"start_inside", "before_start"}:
                reference = [word for word in reference if float(word["start"]) >= target_end][:64]
            else:
                reference = [word for word in reference if float(word["end"]) <= target_start][-64:]
            state["nearby_candidate_speech"] = speech(reference)
        if review_guidance:
            state["guidance"] = REVIEW_GUIDANCE
        edge[name] = {
            "state": state,
            "bounds": [float(target[0]["start"]), float(target[-1]["end"])],
            **unit_meta.get(name, {}),
            **context_meta,
        }
    desired_start, desired_end = case.get("desired_start"), case.get("desired_end")
    return {
        "id": case["id"],
        "split": case["split"],
        "label": case["label"],
        "evidence": case["evidence"],
        "source_sha256": source_sha256,
        "candidate": [start, end],
        "desired": [desired_start, desired_end],
        "undercut_start_seconds": max(0.0, start - desired_start) if desired_start is not None else None,
        "undercut_end_seconds": max(0.0, desired_end - end) if desired_end is not None else None,
        "overcut_start_seconds": max(0.0, desired_start - start) if desired_start is not None else None,
        "overcut_end_seconds": max(0.0, end - desired_end) if desired_end is not None else None,
        "whole": {
            "state": {
                "guidance": GUIDANCE,
                "original_speech": speech(inside),
                "original_interval": {"start": start, "end": end},
                "original_eligible": True,
                "proposed_speech": "",
                "proposed_eligible": False,
                "start_context": speech(words[max(0, words.index(inside[0]) - 16):words.index(inside[0])]),
                "end_context": speech(words[words.index(inside[-1]) + 1:words.index(inside[-1]) + 17]),
            },
            "approximation": "Word-timed replay; caller context and alternative ranked interval are unavailable.",
        },
        "edges": edge,
    }


def ask(state: dict, questions: dict, cache_path: Path, api_key: str, model: str) -> dict:
    payload_sha256 = hashlib.sha256(
        json.dumps({"state": state, "questions": questions, "model": model}, sort_keys=True).encode()
    ).hexdigest()
    started = time.perf_counter()
    result = jev_review_questions(
        state=state,
        questions=questions,
        url=URL,
        api_key=api_key,
        timeout=60.0,
        cache_path=str(cache_path),
        model=model,
        request_deadline=75.0,
    )
    result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
    result["payload_sha256"] = payload_sha256
    return result


def relation_questions(state: dict[str, str]) -> dict:
    return {
        name: {
            "type": "noul",
            "instructions": {
                "question": spec["question"],
                "target_speech": state["target_speech"],
                "nearby_candidate_speech": state["nearby_candidate_speech"],
                "speech_before": state["before_speech"],
                "speech_after": state["after_speech"],
            },
            "criteria": {"true": spec["true"], "false": spec["false"]},
        }
        for name, spec in RELATION_QUESTIONS.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=MODELS, default="jev-1.13.0")
    parser.add_argument("--split", choices=("train", "control", "holdout", "reserve"))
    parser.add_argument("--variant", choices=("baseline", "risk_role", "timed_unit_role", "directional_unit_role", "coref_unit_role", "structured_relation", "structured_relation_review_guidance", "structured_relation_lean"), default="baseline")
    parser.add_argument("--edge", action="append", choices=("start_inside", "end_inside", "before_start", "after_end", "changed_phrase"))
    parser.add_argument("--send", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    known = {case["id"]: case for case in manifest["cases"]}
    if len(known) != len(manifest["cases"]) or any(case not in known for case in args.case):
        raise ValueError("case IDs must be unique and explicitly allowlisted")
    if len(set(args.case)) != len(args.case):
        raise ValueError("duplicate case ID")
    if args.send and args.split is None:
        raise ValueError("--send requires an explicit --split")
    for case_id in args.case:
        case = known[case_id]
        if re.fullmatch(r"[a-z0-9_]+", case_id) is None:
            raise ValueError("invalid case ID")
        if case.get("split") not in {"train", "control", "holdout", "reserve"}:
            raise ValueError("invalid case split")
        if args.split is not None and case["split"] != args.split:
            raise ValueError("selected case is outside the requested split")
    output = args.output.expanduser().resolve()
    if not output.is_relative_to(Path.home() / "Downloads"):
        raise ValueError("output must be under Downloads")
    output.mkdir(parents=True, exist_ok=True)
    credentials = dotenv_values(Path.home() / ".secrets/minuspodjev.env")
    api_key = (
        os.getenv("TYPESAFE_API_KEY")
        or credentials.get("TYPESAFE_API_KEY")
        or credentials.get("MINUSPODJEV_JEV_API")
    )
    if args.send and not api_key:
        raise ValueError("TYPESAFE_API_KEY is unavailable")
    for case_id in args.case:
        result = build_case(
            known[case_id],
            context_words=16 if args.variant in {"structured_relation", "structured_relation_review_guidance", "structured_relation_lean"} else (64 if args.variant != "baseline" else 12),
            timed_units=args.variant in {"timed_unit_role", "directional_unit_role", "coref_unit_role", "structured_relation", "structured_relation_review_guidance", "structured_relation_lean"},
            directional=args.variant == "directional_unit_role",
            relation=args.variant in {"structured_relation", "structured_relation_review_guidance", "structured_relation_lean"},
            review_guidance=args.variant in {"structured_relation_review_guidance", "structured_relation_lean"},
            selected_edges=set(args.edge) if args.edge else None,
        )
        if args.send:
            cache = output / "responses-cache.json"
            whole = result["whole"]["state"]
            result["whole"]["standalone"] = ask(whole, {"original_ad_only": AD_ONLY}, cache, api_key, args.model)
            if args.variant == "baseline":
                result["whole"]["bundled"] = ask(whole, _comparison_question(False, True), cache, api_key, args.model)
            if args.edge and any(edge not in result["edges"] for edge in args.edge):
                raise ValueError("selected edge is unavailable")
            for name, value in result["edges"].items():
                if args.edge and name not in args.edge:
                    continue
                if args.variant in {"timed_unit_role", "directional_unit_role", "coref_unit_role"}:
                    questions = {"role": COREF_UNIT_ROLE if args.variant == "coref_unit_role" else UNIT_ROLE}
                elif args.variant in {"structured_relation", "structured_relation_review_guidance", "structured_relation_lean"}:
                    questions = relation_questions(value["state"])
                elif args.variant == "risk_role":
                    questions = {"risk": {"type": "noul", "instructions": EDGE_RISK}, "role": EDGE_ROLE}
                else:
                    questions = {"ad": {"type": "noul", "instructions": EDGE_AD},
                                 "show": {"type": "noul", "instructions": EDGE_SHOW}}
                request_state = {"guidance": value["state"]["guidance"]} if args.variant == "structured_relation_lean" else value["state"]
                value["result"] = ask(
                    request_state,
                    questions,
                    cache, api_key, args.model,
                )
        result["variant"] = args.variant
        result["requested_model"] = args.model
        result["documented_resolved_model"] = "jev-1.13.0"
        result["model_documentation"] = "https://docs.typesafe.ai/models"
        destination = output / f"{case_id}.json"
        destination.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{case_id}: {destination}")


if __name__ == "__main__":
    main()
