"""LLM ad-response parsing: JSON extraction, timestamp-mode and segment-id-mode
parsers, and the shared post-parse normalization.

Vendored from MinusPod/src/utils/llm_response.py (extract_json_ads_array and its
helper closure) and MinusPod/src/ad_detector/prompts.py (the parse_* functions,
_normalize_ad, the sponsor/category resolvers, and the field-name helpers).
The only edit is that SponsorService.extract_sponsor_from_reason is called as
the vendored module function extract_sponsor_from_reason; behavior is otherwise
unchanged. Confidence/duration thresholds are copied from MinusPod/src/config.py.
"""
import json
import logging
import re

from .schema import SPONSOR_PRIORITY_FIELDS, repair_segment_category
from .sponsors import (
    INVALID_SPONSOR_VALUES,
    STRUCTURAL_FIELDS,
    SPONSOR_PATTERN_KEYWORDS,
    SPONSOR_MAX_NAME_CHARS,
    REASON_DESCRIPTION_MAX,
    is_sponsor_reasoning_rationale,
    mentions_advertising,
    NOT_AD_CLASSIFICATIONS,
    extract_sponsor_from_reason,
)
from .prompt import truncate, parse_timestamp

logger = logging.getLogger('podcast.claude')

# Confidence / duration thresholds (config.py)
CONFIDENCE_STRING_MAP = {
    'high': 0.95,
    'very high': 0.98,
    'medium': 0.75,
    'moderate': 0.75,
    'low': 0.50,
    'very low': 0.30,
}
LOW_CONFIDENCE = 0.50           # Warn/flag for review
CONTENT_DURATION_THRESHOLD = 120.0  # Segments >= this without evidence are likely content
LOW_EVIDENCE_WARN_THRESHOLD = 60.0  # Warn for segments >= this without evidence


# ========== JSON ad-array extraction (utils/llm_response.py) ==========

def find_json_array_candidates(text: str):
    """Yield each top-level ``[...]`` substring from ``text`` in left-to-right
    order.

    Linear-time single-pass scanner: tracks bracket depth and JSON string
    context (so brackets inside ``"..."`` do not affect depth) and records
    each span where the depth transitions from 0 -> 1 -> 0.
    """
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == '[':
            if depth == 0:
                start = i
            depth += 1
        elif ch == ']':
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start:i + 1]
                    start = -1


# Common preamble patterns that some LLMs emit before JSON.
_PREAMBLE_PATTERNS = [
    r'^(?:Here (?:are|is) (?:the )?(?:detected )?ads?[:\s]*)',
    r'^(?:I (?:found|detected|identified)[^:]*[:\s]*)',
    r'^(?:The following (?:ads|advertisements)[^:]*[:\s]*)',
    r'^(?:Based on (?:my|the) analysis[^:]*[:\s]*)',
    r'^(?:After (?:reviewing|analyzing)[^:]*[:\s]*)',
]


def _strip_preamble(text: str, slug: str | None, episode_id: str | None) -> str:
    cleaned = text.strip()
    for pattern in _PREAMBLE_PATTERNS:
        match = re.match(pattern, cleaned, re.IGNORECASE)
        if match:
            cleaned = cleaned[match.end():].strip()
            logger.debug(
                f"[{slug}:{episode_id}] Removed preamble: '{match.group()[:50]}'"
            )
            break
    return cleaned


_NON_AD_TYPES = {'content', 'show', 'music', 'intro', 'outro'}


def _segment_entries_to_ads(entries):
    """Segments-wrapped responses: an entry with start/end is an ad unless an
    explicit type marks it non-ad (models rarely emit type at all, #631)."""
    ads = []
    for s in entries:
        if not isinstance(s, dict) or 'start' not in s or 'end' not in s:
            continue
        t = str(s.get('type', '')).lower()
        if t and t != 'advertisement' and t in _NON_AD_TYPES:
            continue
        ads.append(s)
    return ads


def extract_json_ads_array(
    response_text: str,
    slug: str | None = None,
    episode_id: str | None = None,
) -> tuple[list | None, str | None]:
    """Extract a JSON array of ad dicts from an LLM's response text.

    Tries 4 strategies in order:
    0. Direct JSON parse (handles various wrapper object structures)
    1. Markdown code block extraction
    2. Bracket-depth scan for top-level JSON arrays
    3. Bracket-delimited fallback (first '[' to last ']')

    Returns (ads_list, extraction_method) or (None, None) if no valid JSON found.
    """
    cleaned_text = _strip_preamble(response_text, slug, episode_id)

    try:
        parsed = json.loads(cleaned_text)
        if isinstance(parsed, list):
            return parsed, "json_array_direct"
        if isinstance(parsed, dict):
            if 'window' in parsed and isinstance(parsed['window'], dict):
                window = parsed['window']
                # Include singular "ad": local models (e.g. qwen2.5 via Ollama)
                # often return {"ad": [...]} instead of {"ads": [...]}.
                for key in ['ads_detected', 'ads', 'ad', 'advertisement_segments',
                            'ads_and_sponsorships', 'segments']:
                    if key in window and isinstance(window[key], list):
                        ads = window[key]
                        if key == 'segments':
                            ads = _segment_entries_to_ads(ads)
                        return ads, f"json_object_window_{key}"
            ad_keys = ['ads', 'ad', 'ads_detected', 'advertisement_segments', 'ads_and_sponsorships']
            for key in ad_keys:
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key], f"json_object_{key}_key"
            if 'segments' in parsed and isinstance(parsed['segments'], list):
                ads = _segment_entries_to_ads(parsed['segments'])
                return ads, "json_object_segments_key"
            _has_start = any('start' in k.lower() for k in parsed)
            _has_end = any('end' in k.lower() and k.lower() != 'endorser' for k in parsed)
            if _has_start and _has_end:
                logger.info(f"[{slug}:{episode_id}] Single ad object detected, wrapping in array")
                return [parsed], "json_object_single_ad"
            return [], "json_object_no_ads"
    except json.JSONDecodeError:
        pass

    code_block_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', response_text)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1)), "markdown_code_block"
        except json.JSONDecodeError:
            pass

    scan_text = response_text[:200_000]
    last_valid_ads = None
    for candidate in find_json_array_candidates(scan_text):
        try:
            potential_ads = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(potential_ads, list):
            if not potential_ads or (isinstance(potential_ads[0], dict)
                                     and 'start' in potential_ads[0]):
                last_valid_ads = potential_ads
    if last_valid_ads is not None:
        return last_valid_ads, "regex_json_array"

    clean_response = re.sub(r'```json\s*', '', response_text)
    clean_response = re.sub(r'```\s*', '', clean_response)
    start_idx = clean_response.find('[')
    end_idx = clean_response.rfind(']') + 1
    if start_idx >= 0 and end_idx > start_idx:
        json_str = clean_response[start_idx:end_idx]
        try:
            return json.loads(json_str), "bracket_fallback"
        except json.JSONDecodeError as e:
            logger.warning(
                f"[{slug}:{episode_id}] Strategy 3 JSON parse failed: {e} "
                f"(length={len(json_str)}, start={json_str[:50]!r}, end={json_str[-50:]!r})"
            )

    salvaged = _salvage_truncated_single_ad(response_text)
    if salvaged is not None:
        logger.info(
            f"[{slug}:{episode_id}] Salvaged truncated single-ad JSON "
            f"(model hit max_tokens mid-response)"
        )
        return [salvaged], "json_object_single_ad_truncated"

    return None, None


_TRUNC_NUMERIC_RE = re.compile(r'"(\w+)"\s*:\s*(-?\d+(?:\.\d+)?)')
_TRUNC_STRING_RE = re.compile(r'"(\w+)"\s*:\s*"([^"]{0,500})')


def _salvage_truncated_single_ad(response_text: str) -> dict | None:
    """Recover a usable ad dict from a response that started as a single-ad
    JSON object but ran out of token budget mid-response.

    Observed in the offline benchmark with models that emit verbose ``reason``
    fields and hit ``max_tokens`` before closing the object (Microsoft phi-4
    was the canonical case). The four upstream strategies all bail because
    the JSON is structurally invalid: no closing brace, often an unclosed
    string literal at the very end. We regex-extract the numeric fields and
    optionally the partial reason, then return the dict only if both ``start``
    and ``end`` were recovered. Without those two, downstream IoU matching
    can't do anything with the row, so we'd rather report no-ad than fabricate.
    """
    text = response_text.lstrip()
    # Reviewer + detector prompts return an array of ad verdicts; truncation
    # lands inside the first object. Strip the array wrapper so the salvage
    # regex passes can recover start/end/reason from the partial body.
    if text.startswith("["):
        text = text[1:].lstrip()
    if not text.startswith("{"):
        return None
    fields: dict = {}
    for m in _TRUNC_NUMERIC_RE.finditer(text):
        key, raw = m.group(1), m.group(2)
        if key in ("start", "end", "confidence", "start_time", "end_time",
                  "start_seconds", "end_seconds"):
            try:
                fields[key] = float(raw)
            except ValueError:
                continue
    for m in _TRUNC_STRING_RE.finditer(text):
        key, val = m.group(1), m.group(2)
        if key in ("reason", "sponsor", "advertiser", "end_text", "description"):
            fields.setdefault(key, val)
    start_ok = "start" in fields or "start_time" in fields or "start_seconds" in fields
    end_ok = "end" in fields or "end_time" in fields or "end_seconds" in fields
    if start_ok and end_ok:
        return fields
    return None



# ========== Field-name helpers (ad_detector/prompts.py) ==========

# Two texts can only be duplicates when their lengths are comparable; below
# this ratio the shorter one is a fragment of the longer, which is worth
# keeping rather than discarding.
DUPLICATE_MIN_LENGTH_RATIO = 0.8

def _singular(key: str) -> str:
    """Drop one trailing plural. rstrip('s') stemmed 'names' to 'name' but also
    'address' to 'addre'."""
    lowered = key.lower()
    return lowered[:-1] if lowered.endswith('s') else lowered


# Sponsor field names with a trailing plural dropped, so the evidence gate
# accepts the "sponsors" key extract_sponsor_name already reads.
_SPONSOR_FIELD_STEMS = frozenset(_singular(f) for f in SPONSOR_PRIORITY_FIELDS)


# ========== Category resolver (ad_detector/prompts.py) ==========

# Spelled-out forms of the exact vocabulary: the model reaches for
# "self-promotion" as readily as "self_promo". A position word like "pre-roll"
# or a bare "ad" is still refused.
# Keys a category can arrive under. Only Anthropic enforces the schema, so on
# every other provider the model names fields freely; the rest of this parser
# already tolerates that for start, end and sponsor.
_CATEGORY_KEY_HINTS = ('categor', 'segment_type', 'classification', 'type')


# Shared with sanitize_sponsor_label, which rejects a category name in the
# sponsor slot; one vocabulary, one normalizer.
_repair_category = repair_segment_category


def resolve_ad_category(ad: dict):
    """The segment category an ad object carries, wherever it put it.

    "category" first, then the other keys a model uses for the same idea. The
    value is validated against the vocabulary either way, so a `type` of "ad"
    or "advertisement" contributes nothing while a `type` of "self_promo"
    is taken at face value.
    """
    if not isinstance(ad, dict):
        return None
    direct = _repair_category(ad.get('category'))
    if direct:
        return direct
    for key, value in ad.items():
        kl = str(key).lower()
        if kl == 'category' or not any(h in kl for h in _CATEGORY_KEY_HINTS):
            continue
        found = _repair_category(value)
        if found:
            return found
    return None


# ========== Ad normalization and parsers (ad_detector/prompts.py) ==========

def _flatten_ad_envelopes(ads: list) -> list:
    """Flatten ad-break envelopes the model intermittently emits.

    Instead of a flat list of ad objects, the LLM sometimes wraps each break in
    an envelope like ``{"ad_break_index": N, "ads": [ {ad}, {ad} ]}``. Such an
    envelope has no top-level start/end, so the per-ad parser would discard the
    whole break. Expand any dict whose ``ads`` value is a list into its inner ad
    objects; pass everything else through unchanged.
    """
    flat = []
    for item in ads:
        if isinstance(item, dict) and isinstance(item.get('ads'), list):
            flat.extend(inner for inner in item['ads'] if isinstance(inner, dict))
        else:
            flat.append(item)
    return flat


# The window prompt asks for these notes, so they arrive as prose glued to the
# front of a description and end up in the marker a reader sees.
_CONTINUATION_PREFIX_RE = re.compile(
    r'^(?:continues?\s+(?:from\s+previous|in\s+next)|continued)\b[\s;,.:-]*',
    re.IGNORECASE)


def _drop_leading(description: str, sponsor: str) -> str:
    """Drop a leading sponsor name from a description, so combining the two
    does not render "Acme: Acme ad for ...". Only on a word boundary: "Box"
    must not turn "Boxing gloves" into "ing gloves"."""
    if not description.lower().startswith(sponsor.lower()):
        return description
    rest = description[len(sponsor):]
    if rest and rest[0].isalnum():
        return description
    return rest.lstrip(' :,-.').strip() or description


def _strip_continuation_prefix(text: str) -> str:
    """Drop a leading window-continuation note from description text."""
    return _CONTINUATION_PREFIX_RE.sub('', text or '').lstrip()


def _flatten(value) -> str:
    """Flatten an LLM field to text. A back-to-back break makes the model
    answer a string field with a list, and str() would store the Python repr."""
    if isinstance(value, (list, tuple)):
        return ', '.join(str(v).strip() for v in value if v and str(v).strip())
    return str(value).strip() if value else ''


def _as_text(value) -> str:
    """Flattened text with any leading window-continuation note dropped."""
    return _strip_continuation_prefix(_flatten(value))


def _get_valid_sponsor_value(value):
    if not value:
        return None
    str_value = _flatten(value)
    # A continuation note is window bookkeeping, not a sponsor; trimming
    # the prefix and keeping the remainder minted labels like 'window'.
    if _CONTINUATION_PREFIX_RE.match(str_value):
        return None
    if len(str_value) < 2:
        return None
    if str_value.lower() in INVALID_SPONSOR_VALUES:
        return None
    if len(str_value) > SPONSOR_MAX_NAME_CHARS:
        return None
    if is_sponsor_reasoning_rationale(str_value):
        return None
    return str_value


def _text_is_duplicate(a: str, b: str) -> bool:
    """Check if two strings are essentially the same text.

    Length has to be comparable first. A bare sponsor name is both a
    prefix of its own description and a full word subset of it, so
    without this "Box" swallowed the note explaining the read and left
    the marker saying only "Box".
    """
    a_lower = a.lower().strip()
    b_lower = b.lower().strip()
    shorter, longer = sorted((a_lower, b_lower), key=len)
    if not shorter or len(shorter) < len(longer) * DUPLICATE_MIN_LENGTH_RATIO:
        return False
    if longer.startswith(shorter):
        return True
    a_words = set(a_lower.split())
    b_words = set(b_lower.split())
    if not a_words or not b_words:
        return False
    overlap = len(a_words & b_words)
    smaller = min(len(a_words), len(b_words))
    return overlap / smaller > 0.8 if smaller > 0 else False


def _extract_sponsor_name(ad: dict) -> str:
    """Extract sponsor/advertiser name using priority fields, keywords, and dynamic scanning."""
    # Local alias for the SponsorService method - keeps call sites below short.
    extract_sponsor_from_text = extract_sponsor_from_reason

    for field in SPONSOR_PRIORITY_FIELDS:
        value = _get_valid_sponsor_value(ad.get(field))
        if value:
            return value

    for key in ad.keys():
        key_lower = key.lower()
        for keyword in SPONSOR_PATTERN_KEYWORDS:
            if keyword in key_lower:
                value = _get_valid_sponsor_value(ad.get(key))
                if value:
                    return value

    priority_lower = {f.lower() for f in SPONSOR_PRIORITY_FIELDS}
    for key, val in ad.items():
        key_lower = key.lower()
        if key_lower in STRUCTURAL_FIELDS or key_lower in priority_lower:
            continue
        if isinstance(val, str) and len(val) < 80:
            value = _get_valid_sponsor_value(val)
            if value:
                return value

    for key, val in ad.items():
        if key.lower() in STRUCTURAL_FIELDS:
            continue
        if isinstance(val, str) and len(val) > 10:
            sponsor = extract_sponsor_from_text(val)
            if sponsor:
                return sponsor

    return 'Advertisement detected'


def _normalize_ad(ad: dict, start: float, end: float, slug: str = None,
                   episode_id: str = None, sponsor_service=None) -> dict | None:
    """Post-parse normalization shared by the timestamp-mode and segment-id-mode
    parsers: degenerate-range rejection, is_ad/classification filters, sponsor
    name + reason/description extraction, confidence normalization, the
    duration/evidence gate, and category resolution. ``ad`` still carries all
    raw LLM fields; ``start``/``end`` are already-resolved seconds (parsed
    from timestamp fields, or mapped from segment ids). Returns the final ad
    dict, or None if the candidate is rejected.
    """
    if end <= start:
        logger.warning(
            f"[{slug}:{episode_id}] Discarding ad candidate: "
            f"invalid range (start={start:.1f}s >= end={end:.1f}s) - "
            f"reason={str(ad.get('reason', ad.get('sponsor', '')))[:80]}"
        )
        return None

    # Filter out explicitly marked non-ads
    is_ad_val = ad.get('is_ad')
    if is_ad_val is not None:
        if str(is_ad_val).lower() in ('false', 'no', '0', 'none'):
            logger.info(f"[{slug}:{episode_id}] Skipping non-ad: "
                        f"{start:.1f}s-{end:.1f}s (is_ad={is_ad_val})")
            return None

    # Filter by classification/type field
    classification = str(ad.get('classification') or ad.get('type') or '').lower()
    if classification in NOT_AD_CLASSIFICATIONS:
        logger.info(f"[{slug}:{episode_id}] Skipping non-ad: "
                    f"{start:.1f}s-{end:.1f}s (classification={classification})")
        return None

    # Extract sponsor/advertiser name using priority fields + pattern matching
    # Try extract_sponsor_name first for a real sponsor name.
    # If it returns the default, fall back to Claude's raw reason.
    sponsor_name = _extract_sponsor_name(ad)
    reason = sponsor_name
    existing_reason = ad.get('reason')
    if reason == 'Advertisement detected':
        if existing_reason and isinstance(existing_reason, str) and len(existing_reason) > 3:
            reason = existing_reason
    elif existing_reason and isinstance(existing_reason, str) and len(existing_reason) > len(reason) + 5:
        # Claude's reason is substantially more descriptive than the bare sponsor name
        reason = existing_reason

    # Extract description from Claude's response to enrich the reason
    # Dynamic scan: check ALL non-structural string fields > 10 chars
    # Skip 'reason' (already used above); duplication with sponsor handled at combine time
    description = None
    for key, val in ad.items():
        if key.lower() in STRUCTURAL_FIELDS:
            continue
        if key == 'reason':
            continue
        if isinstance(val, str) and len(val) > 10:
            # Prefer longer descriptive text over short values
            if description is None or len(val) > len(description):
                description = val
    # Kept whole (#591); the old 300/150 caps put a literal
    # "..." in the UI with no fuller text to expand to.
    description = truncate(
        _strip_continuation_prefix(description),
        REASON_DESCRIPTION_MAX)

    # Combine sponsor + description in reason field
    if description:
        if reason and reason != 'Advertisement detected':
            # Avoid duplication: check if description is essentially the same text
            if not _text_is_duplicate(reason, description):
                description = _drop_leading(description, reason)
                reason = f"{reason}: {description}" if description else reason
        elif not reason or reason == 'Advertisement detected':
            reason = description

    # Normalize confidence to 0-1 range
    raw_conf = ad.get('confidence', 0.8)
    if isinstance(raw_conf, str):
        mapped = CONFIDENCE_STRING_MAP.get(raw_conf.lower().strip())
        if mapped is not None:
            logger.debug(f"[{slug}:{episode_id}] Mapped string confidence '{raw_conf}' -> {mapped}")
            raw_conf = mapped
        else:
            raw_conf = raw_conf.rstrip('%')
    raw_conf = float(raw_conf)
    norm_conf = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf
    norm_conf = min(1.0, max(0.0, norm_conf))

    # Dynamic validation: require positive evidence this is an ad
    # instead of blocklisting content indicators (which keeps growing)
    duration = end - start
    has_sponsor_field = any(
        _singular(key) in _SPONSOR_FIELD_STEMS
        and _get_valid_sponsor_value(val)
        for key, val in ad.items()
    )
    has_known_sponsor = (
        sponsor_service and
        sponsor_service.find_sponsor_in_text(reason)
    ) if reason else False
    has_ad_language = mentions_advertising(reason)

    if not has_sponsor_field and not has_known_sponsor and not has_ad_language:
        # Low confidence + no evidence = reject regardless of duration
        if norm_conf < LOW_CONFIDENCE:
            logger.info(
                f"[{slug}:{episode_id}] Rejecting low-confidence non-sponsor: "
                f"{start:.1f}s-{end:.1f}s ({duration:.0f}s, conf={norm_conf:.0%}) - "
                f"reason: {reason[:100] if reason else 'None'}"
            )
            return None
        # No positive ad evidence -- apply duration gate
        # Short segments (<CONTENT_DURATION_THRESHOLD) get benefit of doubt
        # Long segments are almost certainly content descriptions
        if duration >= CONTENT_DURATION_THRESHOLD:
            logger.info(
                f"[{slug}:{episode_id}] Rejecting suspected content: "
                f"{start:.1f}s-{end:.1f}s ({duration:.0f}s) - "
                f"no sponsor identified in reason: {reason[:100] if reason else 'None'}"
            )
            return None
        # For shorter segments without evidence, log warning but allow through
        if duration >= LOW_EVIDENCE_WARN_THRESHOLD:
            logger.warning(
                f"[{slug}:{episode_id}] Low-confidence ad (no sponsor found): "
                f"{start:.1f}s-{end:.1f}s ({duration:.0f}s) - "
                f"reason: {reason[:100] if reason else 'None'}"
            )

    logger.info(f"[{slug}:{episode_id}] Extracted ad: {start:.1f}s-{end:.1f}s, reason='{reason}', fields={list(ad.keys())}")
    ad_entry = {
        'start': start,
        'end': end,
        'confidence': norm_conf,
        'reason': reason,
        'end_text': _as_text(ad.get('end_text'))
    }
    # Store sponsor name separately for UI display
    # (reuses sponsor_name captured above; ad is unmutated between)
    if sponsor_name and sponsor_name != 'Advertisement detected':
        ad_entry['sponsor'] = sponsor_name
    # Pass the LLM's raw category through unvalidated; the
    # merge seam normalizes it against SEGMENT_CATEGORIES.
    resolved_category = resolve_ad_category(ad)
    if resolved_category:
        ad_entry['category'] = resolved_category
    return ad_entry


def parse_ads_from_response(response_text: str, slug: str = None,
                              episode_id: str = None,
                              sponsor_service=None,
                              compliance_meta: dict | None = None) -> list[dict]:
    """Parse ad segments from Claude's JSON response.

    ``compliance_meta``: optional out-param dict (same pattern as
    ``run_stats`` elsewhere in the codebase). When given, this sets
    ``compliance_meta['extraction_failed']`` to True when no JSON ads array
    could be located or parsed at all, and False when a JSON array was
    successfully parsed -- including a valid, empty ``[]`` answer. Used by
    the addressing-mode compliance stats (timestamps effective mode) to
    distinguish "the model answered with no ads" from "the response wasn't
    parseable".

    Returns:
        List of validated ad dicts with start, end, confidence, reason, end_text
    """
    try:
        ads, extraction_method = extract_json_ads_array(response_text, slug, episode_id)

        if ads is None or not isinstance(ads, list):
            logger.warning(f"[{slug}:{episode_id}] No valid JSON array found in response")
            if compliance_meta is not None:
                compliance_meta['extraction_failed'] = True
            return []
        if compliance_meta is not None:
            compliance_meta['extraction_failed'] = False

        # Flatten any {"ad_break_index": N, "ads": [...]} envelopes the model
        # sometimes emits, so the per-ad parser below sees the inner ad objects.
        ads = _flatten_ad_envelopes(ads)

        # Validate and normalize ads - handle various field name patterns
        valid_ads = []
        for ad in ads:
            if isinstance(ad, dict):
                # Log raw ad object for debugging
                logger.debug(f"[{slug}:{episode_id}] Raw ad from LLM: {json.dumps(ad, default=str)[:500]}")
                # Fuzzy-match start/end timestamp fields from LLM response.
                # The LLM uses inconsistent field names across runs (start_time,
                # ad_start, timestamp_start, start_time_seconds, etc). Instead of
                # maintaining an ever-growing allowlist, match any key containing
                # 'start'/'end' that isn't a known text field.
                _SKIP_SUFFIXES = ('_note', '_text', '_snip', '_quote', '_description')
                _SKIP_KEYS = {'endorser', 'endorsed', 'price_starting', 'starting_point'}
                start_val = None
                end_val = None
                for k, v in ad.items():
                    kl = k.lower()
                    if kl in _SKIP_KEYS or any(kl.endswith(s) for s in _SKIP_SUFFIXES):
                        continue
                    if v is None:
                        continue
                    if start_val is None and 'start' in kl:
                        start_val = v
                    elif end_val is None and 'end' in kl and kl != 'endorser':
                        end_val = v

                if start_val is None or end_val is None:
                    logger.warning(
                        f"[{slug}:{episode_id}] Discarding ad candidate: "
                        f"missing timestamps (start={start_val}, end={end_val}) - "
                        f"fields={list(ad.keys())}, reason={str(ad.get('reason', ad.get('sponsor', '')))[:80]}"
                    )
                    continue

                try:
                    start = parse_timestamp(start_val)
                    end = parse_timestamp(end_val)
                    ad_entry = _normalize_ad(
                        ad, start, end, slug, episode_id, sponsor_service)
                    if ad_entry is not None:
                        valid_ads.append(ad_entry)
                except ValueError as e:
                    logger.warning(f"[{slug}:{episode_id}] Skipping ad with invalid timestamp: {e}")
                    continue

        return valid_ads

    except json.JSONDecodeError as e:
        logger.error(f"[{slug}:{episode_id}] Failed to parse JSON: {e}")
        if compliance_meta is not None:
            compliance_meta['extraction_failed'] = True
        return []


def _int_field(obj: dict, keys: tuple[str, ...]):
    """The first of ``keys`` present in ``obj``, coerced to int, or None if
    absent or non-numeric. Exact key match only -- unlike the fuzzy
    'start'/'end' substring matcher in ``parse_ads_from_response``, id fields
    must never be guessed at, or ``start_id`` could be misread by a substring
    match on 'start' as a timestamp."""
    for key in keys:
        if key in obj:
            try:
                return int(obj[key])
            except (TypeError, ValueError):
                return None
    return None


def parse_id_ads_from_response(response_text: str, slug: str = None,
                                episode_id: str = None,
                                sponsor_service=None) -> tuple[list[dict], bool]:
    """Parse an ID-mode LLM response (issue: hushpod adoption).

    Returns (ads, used_ids). used_ids is True when at least one object in
    the response carries integer id fields; ads then hold 'start_id'/'end_id'
    plus the usual fields (confidence, category, reason -- normalized later
    by ``resolve_segment_id_ads``). used_ids False means the model ignored
    the ID contract (some models do): the caller re-parses with
    ``parse_ads_from_response`` so the window is not lost, at the cost of
    approximate timestamps for that window.
    """
    try:
        raw, _extraction_method = extract_json_ads_array(response_text, slug, episode_id)
    except Exception as e:
        logger.warning(f"[{slug}:{episode_id}] Failed to extract ID-mode ads: {e}")
        return [], False
    if raw is None or not isinstance(raw, list):
        return [], False
    raw = _flatten_ad_envelopes(raw)
    if not raw:
        return [], True  # explicit empty "no ads" answer is a valid ID answer

    ads = []
    any_ids = False
    skipped_no_id = 0
    for obj in raw:
        if not isinstance(obj, dict):
            continue
        sid_lo = _int_field(obj, ('start_id', 'startid', 'start_segment_id'))
        sid_hi = _int_field(obj, ('end_id', 'endid', 'end_segment_id'))
        if sid_lo is None or sid_hi is None:
            skipped_no_id += 1
            continue
        any_ids = True
        ad = dict(obj)
        ad['start_id'], ad['end_id'] = sid_lo, sid_hi
        # An id-mode object should never carry timestamp fields, but strip
        # them defensively so a stray 'start'/'end' can't leak through
        # resolve_segment_id_ads and be mistaken for the resolved value.
        ad.pop('start', None)
        ad.pop('end', None)
        ads.append(ad)
    if any_ids and skipped_no_id:
        # Mixed response: some objects used the id contract, others didn't
        # (likely timestamp-mode ads in the same response). The id-less
        # objects are silently dropped below -- surface the count so a lost
        # detection is diagnosable instead of invisible.
        logger.warning(
            f"[{slug}:{episode_id}] ID-mode response mixed formats: "
            f"skipped {skipped_no_id} object(s) without id fields")
    return (ads, True) if any_ids else ([], False)


def resolve_segment_id_ads(ads: list[dict], window_segments: list[dict],
                            slug: str = None, episode_id: str = None,
                            sponsor_service=None) -> list[dict]:
    """Map start_id/end_id to exact segment start/end seconds, then run the
    resolved ads through the same post-parse normalization
    ``parse_ads_from_response`` applies (confidence normalization, sponsor
    extraction, degenerate-range rejection, evidence gate, category
    resolution) via ``_normalize_ad``.

    Unknown ids drop the detection (an invented id is detectable; an
    invented timestamp is not -- that asymmetry is the point of this mode).
    """
    by_sid = {seg['sid']: seg for seg in window_segments if 'sid' in seg}
    resolved = []
    for ad in ads:
        lo = min(ad['start_id'], ad['end_id'])
        hi = max(ad['start_id'], ad['end_id'])
        seg_lo, seg_hi = by_sid.get(lo), by_sid.get(hi)
        if seg_lo is None or seg_hi is None:
            logger.warning(
                f"[{slug}:{episode_id}] Dropping detection with out-of-window "
                f"segment ids {lo}-{hi}")
            continue
        raw = {k: v for k, v in ad.items() if k not in ('start_id', 'end_id')}
        start = seg_lo['start']
        end = seg_hi['end']
        try:
            ad_entry = _normalize_ad(raw, start, end, slug, episode_id, sponsor_service)
        except (ValueError, TypeError) as e:
            logger.warning(
                f"[{slug}:{episode_id}] Skipping ad with invalid field "
                f"(ids {lo}-{hi}): {e}")
            continue
        if ad_entry is not None:
            resolved.append(ad_entry)
    return resolved
