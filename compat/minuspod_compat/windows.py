"""Transcript window framing and the per-stage tunable resolver create_windows
relies on.

Vendored from MinusPod/src/ad_detector/prompts.py (create_windows) and
MinusPod/src/config.py (WINDOW_SIZE_SECONDS, WINDOW_OVERLAP_SECONDS, the
STAGE_TUNABLE_* window entries, _coerce_tunable, get_stage_tunable).

get_stage_tunable is copied faithfully except that the DB/cache lookup (which
in MinusPod read llm_client._get_cached_setting) is dropped: this standalone
package has no settings database, so a caller-supplied ``settings`` dict is the
only DB-tier source and resolution otherwise falls through env var then default.
Only the two detection-window keys create_windows reads are kept in the tunable
tables; no other stage tunable is referenced here.
"""
import logging
import os
from typing import Any

logger = logging.getLogger('podcast.claude')
_tunable_logger = logging.getLogger(__name__)

# detection window geometry defaults (config.py)
WINDOW_SIZE_SECONDS = 600       # Claude processing window (10 min)
WINDOW_OVERLAP_SECONDS = 180    # Overlap between windows (3 min)

STAGE_TUNABLE_DEFAULTS = {
    # detection window geometry (global, not per-stage)
    'window_size_seconds': WINDOW_SIZE_SECONDS,
    'window_overlap_seconds': WINDOW_OVERLAP_SECONDS,
}

STAGE_TUNABLE_ENV_VARS = {key: key.upper() for key in STAGE_TUNABLE_DEFAULTS}

# Legacy env-var aliases for backward compatibility. (None apply to the window
# keys; kept for parity with the resolver's lookup.)
STAGE_TUNABLE_ENV_ALIASES = {}

STAGE_TUNABLE_RANGES = {
    # detection window geometry. Cross-field constraint (overlap < size) is
    # enforced at the API layer; the per-field bounds here are the static
    # envelope the resolver checks against.
    'window_size_seconds': (120, 10800),
    'window_overlap_seconds': (0, 1770),
}

STAGE_TUNABLE_REASONING_LEVELS = {"none", "low", "medium", "high"}


def _coerce_tunable(key: str, raw: Any, source_label: str) -> Any | None:
    """Coerce a stored or env value to int/float/enum. Returns None on bad value
    (caller treats None as 'use default')."""
    if raw is None:
        return None
    raw_str = str(raw).strip()
    if raw_str == "":
        return None

    if key.endswith('_reasoning_level'):
        normalized = raw_str.lower()
        if normalized in STAGE_TUNABLE_REASONING_LEVELS:
            return normalized
        _tunable_logger.warning(
            f"{source_label}={raw!r} is not a valid reasoning level; using default"
        )
        return None

    range_ = STAGE_TUNABLE_RANGES.get(key)
    if key.endswith('_temperature'):
        try:
            v: Any = float(raw_str)
        except ValueError:
            _tunable_logger.warning(f"{source_label}={raw!r} is not numeric; using default")
            return None
    else:
        try:
            v = int(raw_str)
        except ValueError:
            _tunable_logger.warning(f"{source_label}={raw!r} is not an integer; using default")
            return None

    if range_ is not None:
        lo, hi = range_
        if not (lo <= v <= hi):
            _tunable_logger.warning(
                f"{source_label}={v} is out of range [{lo}, {hi}]; using default"
            )
            return None
    return v



def get_stage_tunable(key: str, settings: dict | None = None) -> Any:
    """Resolve DB > env > default for a per-stage tunable.

    Same precedence as every other env-backed setting: a value saved in the
    Settings UI wins, the env var supplies the default when no UI value
    exists (issue #491 consolidation; env used to win here).

    Out-of-range or malformed values produce a WARNING and the next source
    is used, never an exception, so a bad value in the DB never blocks
    processing.

    Args:
        key: Tunable key.
        settings: Pre-loaded {key: {'value', 'is_default'}} dict from
            db.get_all_settings(). When supplied, skips the DB read -- used
            by the Settings GET handler to resolve all tunables off the
            single query it already issues. When omitted, the lookup hits
            the shared 5s TTL cache in llm_client so per-window calls during
            episode processing don't issue a fresh SQLite read each pass.
    """
    if key not in STAGE_TUNABLE_DEFAULTS:
        raise KeyError(f"Unknown stage tunable: {key!r}")
    default = STAGE_TUNABLE_DEFAULTS[key]

    # DB lookup first. Caller-supplied dict takes precedence; otherwise use
    # the shared TTL cache so stage code calling this on every window doesn't
    # hammer SQLite. 5s TTL still propagates Settings UI changes promptly.
    db_val: str | None = None
    if settings is not None:
        entry = settings.get(key)
        if isinstance(entry, dict):
            db_val = entry.get('value')
        elif hasattr(entry, 'value'):
            # SettingEntry dataclass (api.settings._settings_view wraps the
            # raw dict shape into typed entries). hasattr check keeps the
            # resolver pluggable to other entry shapes consumers add.
            db_val = entry.value
        else:
            db_val = entry
    else:
        # Standalone package: no settings database or cache. Fall through to
        # the env var and then the default.
        db_val = None

    if db_val is not None and str(db_val).strip() != "":
        coerced = _coerce_tunable(key, db_val, f"settings[{key}]")
        if coerced is not None:
            return coerced

    env_name = STAGE_TUNABLE_ENV_VARS[key]
    env_val = os.environ.get(env_name)
    used_env = env_name
    if env_val is None:
        alias = STAGE_TUNABLE_ENV_ALIASES.get(key)
        if alias:
            env_val = os.environ.get(alias)
            if env_val is not None:
                used_env = alias
    if env_val is not None and env_val.strip() != "":
        coerced = _coerce_tunable(key, env_val, used_env)
        return coerced if coerced is not None else default

    return default


def create_windows(segments: list[dict], window_size: float = None,
                   overlap: float = None) -> list[dict]:
    """Create overlapping windows from transcript segments.

    Args:
        segments: List of transcript segments with 'start', 'end', 'text'
        window_size: Duration of each window in seconds. None resolves the
            user-configurable 'window_size_seconds' tunable at call time so
            Settings UI changes take effect without restart.
        overlap: Overlap between consecutive windows in seconds. None resolves
            the user-configurable 'window_overlap_seconds' tunable at call time.

    Returns:
        List of window dicts with:
            - 'start': window start time (absolute)
            - 'end': window end time (absolute)
            - 'segments': list of segments in this window
    """
    if not segments:
        return []

    if window_size is None:
        window_size = get_stage_tunable('window_size_seconds')
    if overlap is None:
        overlap = get_stage_tunable('window_overlap_seconds')

    # Get total transcript duration
    total_duration = segments[-1]['end']
    step_size = window_size - overlap
    if step_size <= 0:
        # overlap >= window_size never advances window_start and hangs the
        # detection worker forever. The Settings API cross-field check can be
        # bypassed via env vars or a direct DB write, so guard here too: fall
        # back to non-overlapping windows rather than wedge (config-1 /
        # ad-detection-1).
        logger.warning(
            "create_windows: window_overlap_seconds (%s) >= window_size_seconds "
            "(%s); falling back to non-overlapping windows to avoid a "
            "non-terminating loop.", overlap, window_size,
        )
        step_size = max(window_size, 1.0)

    windows = []
    window_start = 0.0

    while window_start < total_duration:
        window_end = min(window_start + window_size, total_duration)

        # Find segments that overlap with this window
        window_segments = []
        for seg in segments:
            # Segment overlaps if it starts before window ends AND ends after window starts
            if seg['start'] < window_end and seg['end'] > window_start:
                window_segments.append(seg)

        if window_segments:
            windows.append({
                'start': window_start,
                'end': window_end,
                'segments': window_segments
            })

        window_start += step_size

    logger.debug(f"Created {len(windows)} windows from {total_duration/60:.1f} min transcript")
    return windows
