"""Prompt templates, placeholder substitution, window/system prompt builders,
and the small text/time helpers the parser needs.

Vendored from MinusPod/src/utils/prompt.py (render_prompt, format_sponsor_block,
strip_comments_from_prompt, strip_html, scrub_description),
MinusPod/src/utils/text.py (truncate),
MinusPod/src/utils/time.py (parse_timestamp, utc_now_iso, format_time), and
MinusPod/src/ad_detector/prompts.py (USER_PROMPT_TEMPLATE, format_window_prompt,
SEGMENT_ID_SYSTEM_SECTION, SEGMENT_ID_WINDOW_RULES, get_static_system_prompt).
Behavior unchanged from the originals.
"""
import re
from datetime import datetime, timezone


# ========== Placeholder substitution (utils/prompt.py) ==========

SPONSOR_DATABASE_HEADER = (
    "\n\nDYNAMIC SPONSOR DATABASE (current known sponsors - treat as high confidence):\n"
)


def render_prompt(prompt: str, **vars: str) -> str:
    """Substitute ``{name}`` placeholders in ``prompt`` with provided values.

    Variables without a corresponding placeholder are silently dropped: that
    is the supported way for a user to opt out of an injection by removing
    the placeholder from their customized prompt.
    """
    rendered = prompt
    for name, value in vars.items():
        rendered = rendered.replace('{' + name + '}', value)
    return rendered


def format_sponsor_block(sponsor_list: str) -> str:
    """Wrap a non-empty sponsor list with the standard header.

    Empty list returns empty string so substitution does not produce a
    dangling header on prompts whose ``{sponsor_database}`` placeholder is
    left in place.
    """
    if not sponsor_list:
        return ""
    return SPONSOR_DATABASE_HEADER + sponsor_list


def strip_comments_from_prompt(prompt: str|None) -> str|None:
    """Remove HTML-style comments from the prompt following markdown
    conventions.

    Multi-line comments can start at the beginning of a line (up to three
    leading spaces allowed), and single-line comments can appear anywhere.
    If a ML comment ends with a line break, the line break is also removed.

    HTML-style comments cannot be nested (the nested comment's end will
    terminate the outer comment).

    Examples:
        Fooo <!-- This is a single-line comment -->
        <!--
        This is a multi-line comment
        -->
    """
    if not prompt:
        return prompt
    # A comment indented four or more spaces is markdown code and stays.
    pattern = (r'^([ ]{0,3})<!--.*?-->(?:\r?\n)?'
               r'|^([ ]{4,}<!--.*?-->)'
               r'|<!--.*?-->')
    return re.sub(pattern, lambda m: m.group(1) or m.group(2) or '',
                  prompt, flags=re.MULTILINE | re.DOTALL)


# ========== Text helper (utils/text.py) ==========

def truncate(text: str, limit: int) -> str:
    """Cut text to limit characters, ellipsis included in the count."""
    if not text or len(text) <= limit:
        return text
    # No room for the ellipsis: text[:limit - 3] would slice from the end.
    if limit <= 3:
        return text[:max(limit, 0)]
    return text[:limit - 3].rstrip() + '...'




# ========== HTML/description scrub (utils/prompt.py) ==========

def strip_html(text: str|None) -> str|None:
    """Convert simple HTML to plain text for show-note timestamp parsing.

    Block-level tags must be turned into newlines (not just stripped) so the
    downstream `_TIMESTAMP_PATTERNS` regex sees each timestamp on its own line.
    A bare tag-stripper like nh3 would collapse `<p>00:00 A</p><p>05:30 B</p>`
    into `00:00 A05:30 B` and miss every anchor after the first.
    """
    if not text:
        return text
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</(p|li|div)>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    for entity, char in (('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'),
                         ('&quot;', '"'), ('&#39;', "'"), ('&nbsp;', ' ')):
        text = text.replace(entity, char)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def scrub_description(description: str|None, max_length: int = 800) -> str:
    """Scrub the description of HTML, timestamps, URLs, excessive whitespace,
    and then truncate to the first `max_length` characters, splitting on a
    word boundary and appending ellipsis if needed.
    """
    if not description:
        return ""
    if max_length <= 0:
        return "..."
    description = strip_html(description)
    # replace timestamps with 'XX:XX' to avoid hallucinations
    description = re.sub(r'(?:\d+:)?\d{1,2}:\d{2}', 'XX:XX', description)
    # shorten urls (keep scheme and domain)
    description = re.sub(r'(https?://[^/\s]+)/\S+', r'\1/...', description)
    # remove trailing whitespace and empty lines
    description = re.sub(r'(?:\s*\n)+', '\n', description)
    if len(description) > max_length:
        description = description[:max_length]
        last_space = description.rfind(' ')
        if last_space > 0:
            description = description[:last_space]
        description += "..."
    return description


# ========== Time helper (utils/time.py) ==========

ISO_FORMAT = '%Y-%m-%dT%H:%M:%SZ'


def utc_now_iso() -> str:
    """Return current UTC time as ISO 8601 string (e.g. '2026-03-15T12:00:00Z')."""
    return datetime.now(timezone.utc).strftime(ISO_FORMAT)


def format_time(seconds: float, include_hours: bool = False) -> str:
    """Format seconds as human-readable timestamp string.

    Returns:
        Formatted timestamp (H:MM:SS.ss or M:SS.ss)
    """
    if seconds < 0:
        seconds = 0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if hours > 0 or include_hours:
        return f"{hours}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes}:{secs:05.2f}"


def parse_timestamp(ts) -> float:
    """Convert timestamp value to seconds.

    Supports multiple input types and formats:
    - int/float: passed through directly (e.g., 1178.5 -> 1178.5)
    - String with 's' suffix: "1178.5s" -> 1178.5
    - Float string: "1178.5" -> 1178.5
    - HH:MM:SS.mmm (e.g., "01:23:45.678")
    - HH:MM:SS (e.g., "01:23:45")
    - MM:SS.mmm (e.g., "23:45.678")
    - MM:SS (e.g., "23:45")
    - M:SS (e.g., "3:45")

    Also handles comma as decimal separator (common in some VTT files).

    Raises:
        ValueError: If the timestamp cannot be parsed
    """
    if isinstance(ts, (int, float)):
        return float(ts)

    if not ts or not isinstance(ts, str):
        raise ValueError(f"Cannot parse timestamp: {ts!r}")

    # Normalize: strip whitespace, remove 's' suffix, replace comma decimal
    ts = ts.strip().rstrip('s').strip().replace(',', '.')

    # Try direct float conversion first (handles "1178.5" etc.)
    try:
        return float(ts)
    except ValueError:
        pass

    # Try colon-separated formats
    parts = ts.split(':')

    try:
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
    except (ValueError, IndexError):
        pass

    raise ValueError(f"Cannot parse timestamp: {ts!r}")


# ========== Prompt templates and builders (ad_detector/prompts.py) ==========

# User prompt template (not configurable via UI - just formats the transcript)
# Description is optional - may contain sponsor lists, chapter markers, or content context
USER_PROMPT_TEMPLATE = """Podcast: {podcast_name}
Episode: {episode_title}
{description_section}
Transcript:
{transcript}"""


def format_window_prompt(
    podcast_name: str,
    episode_title: str,
    description_section: str,
    transcript_lines: list[str],
    window_index: int,
    total_windows: int,
    window_start: float,
    window_end: float,
    audio_context: str = "",
    addressing_mode: str = "timestamps",
) -> str:
    """Build the user prompt for a single ad-detection window.

    `description_section` and `audio_context` are pre-built strings so the
    benchmark can call this without DB or audio-analysis state. Production
    callers assemble both then pass them in.

    `addressing_mode` selects the window-context rules appended after the
    header: 'timestamps' (default) emits the original three bullets telling
    the model to use absolute timestamps, byte-identical to before this
    parameter existed; 'segment_ids' (issue: hushpod adoption) emits
    SEGMENT_ID_WINDOW_RULES instead of those bullets -- never both, so the
    prompt never tells the model to use and never use timestamps in the same
    message. Callers that never pass it (benchmark, keep-content windows)
    get 'timestamps' behavior unchanged.
    """
    transcript = "\n".join(transcript_lines)
    header = (
        f"\n\n=== WINDOW {window_index + 1}/{total_windows}: "
        f"{window_start/60:.1f}-{window_end/60:.1f} minutes ==="
    )
    if addressing_mode == "segment_ids":
        rules = SEGMENT_ID_WINDOW_RULES
    else:
        rules = (
            "\n- Use absolute timestamps from transcript (as shown in brackets)"
            "\n- If an ad starts before this window, use the first timestamp with note \"continues from previous\""
            f"\n- If an ad extends past this window, use {window_end:.1f} with note \"continues in next\"\n"
        )
    window_context = header + rules
    return strip_comments_from_prompt(USER_PROMPT_TEMPLATE).format(
        podcast_name=podcast_name,
        episode_title=episode_title,
        description_section=description_section,
        transcript=transcript,
    ) + audio_context + window_context


SEGMENT_ID_SYSTEM_SECTION = """

ADDRESSING MODE: SEGMENT IDS
The transcript is a numbered list; each line starts with its [id]. For every
detection you report, replace the "start" and "end" timestamp fields with
integer "start_id" and "end_id" fields: the ids of the FIRST and LAST
transcript lines of the ad, inclusive. Refer to lines ONLY by the ids shown.
Never output timestamps and never invent ids that do not appear in the
transcript. All other rules (categories, confidence, reason) are unchanged.

Ignore any earlier instruction to read [Xs] timestamp markers or to output
numeric "start"/"end" seconds: in this mode the transcript lines carry [id]
numbers only, and the JSON fields "start"/"end" are replaced by integer
"start_id"/"end_id". All other rules (categories, confidence, reason) still
apply."""


SEGMENT_ID_WINDOW_RULES = (
    "\n- Report start_id/end_id integers from the [id] brackets, "
    "never timestamps"
    "\n- If an ad starts before this window, use this window's first id "
    "with note \"continues from previous\""
    "\n- If an ad extends past this window, use this window's last id "
    "with note \"continues in next\"\n"
)



def get_static_system_prompt() -> str:
    """Return DEFAULT_SYSTEM_PROMPT with the static SEED_SPONSORS list substituted.

    Reproducible from source code -- no DB, env, or wallclock dependency.
    Used by the offline LLM benchmark. Production reads stored prompts and
    merges DB-derived sponsors via ``AdDetector.get_system_prompt`` instead.
    """
    from .sponsors import DEFAULT_SYSTEM_PROMPT
    from .sponsors import SEED_SPONSORS
    sponsor_list = ', '.join(s['name'] for s in SEED_SPONSORS)
    return render_prompt(
        strip_comments_from_prompt(DEFAULT_SYSTEM_PROMPT),
        sponsor_database=format_sponsor_block(sponsor_list),
    )
