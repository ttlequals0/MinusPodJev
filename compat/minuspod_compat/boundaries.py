"""Cross-window ad deduplication/merge.

Vendored from MinusPod/src/ad_detector/boundaries.py (deduplicate_window_ads and
its action helpers resolve_category_action, effective_resolved_action,
split_conflicting_action_span) and MinusPod/src/config.py
(normalize_segment_category, DEFAULT_SEGMENT_ACTION, MIN_AD_DURATION_FOR_REMOVAL).
The merge/split bookkeeping helpers (note_fold, carve_fragment,
clip_dai_core_spans) come from the vendored .markers module. Behavior unchanged
from the originals.
"""
import logging
from typing import Any

from .schema import SEGMENT_CATEGORIES
from .markers import carve_fragment, clip_dai_core_spans, note_fold

logger = logging.getLogger('podcast.claude')

# config.py constants create the action gate and the measured-fragment floor.
DEFAULT_SEGMENT_ACTION = 'remove'
MIN_AD_DURATION_FOR_REMOVAL = 10.0   # Min ad duration to actually remove from audio


def normalize_segment_category(value: Any) -> str:
    """Return value if it is a known segment category, else 'sponsor'.

    For resolving a per-category action, where an unknown category has to
    resolve to something and 'sponsor' is the conservative choice (cut it).
    Do not use this to record or display a category: an unset one is left
    unset so 'sponsor' keeps meaning a real sponsor read.
    """
    return value if value in SEGMENT_CATEGORIES else 'sponsor'



def resolve_category_action(category, action_map: dict[str, str]) -> str:
    """Resolve a marker's category to its feed-configured action.

    Only call with a non-None action_map; callers with no map treat every
    category as the same action and never call this.
    """
    return action_map.get(normalize_segment_category(category), DEFAULT_SEGMENT_ACTION)


def effective_resolved_action(marker: dict,
                              action_map: dict[str, str] | None) -> str | None:
    """Marker's resolved action with the pattern-overrides-keep rule applied,
    matching split_conflicting_action_span's precedence input."""
    if action_map is None:
        return None
    action = resolve_category_action(marker.get('category'), action_map)
    if action == 'keep' and marker.get('pattern_defined'):
        return 'remove'
    return action


def split_conflicting_action_span(last: dict, current: dict,
                                  last_action: str | None = None,
                                  current_action: str | None = None) -> tuple:
    """Resolve two adjacent-or-overlapping ads whose resolved actions differ:
    never merge a keep-resolving detection into a remove-resolving one, and
    never let a span fully nested inside the other collapse to nothing.

    Returns ``(updated_last_or_None, new_entries)``: ``new_entries`` replaces
    ``current`` in the merged list; ``updated_last`` (None when fully
    consumed) replaces ``last``.

    - No true overlap: both survive untouched.
    When actions are supplied, precedence is keep > beep > remove. The
    higher-priority action owns contested audio; without actions, preserve
    the historical behavior where ``current`` owns it.
    """
    if current['start'] >= last['end']:
        return last, [current.copy()]

    def mark_measured_fragment(fragment, parent):
        if (parent.get('_measured_split_fragment')
                or parent['end'] - parent['start']
                >= MIN_AD_DURATION_FOR_REMOVAL):
            fragment['_measured_split_fragment'] = True
        return fragment

    def carve(parent, s, e):
        return mark_measured_fragment(carve_fragment(parent, s, e), parent)

    priority = {'remove': 0, 'beep': 1, 'keep': 2}
    last_pattern = bool(last.get('pattern_defined'))
    current_pattern = bool(current.get('pattern_defined'))
    effective_last_action = (
        'remove' if last_pattern and last_action == 'keep' else last_action)
    effective_current_action = (
        'remove' if current_pattern and current_action == 'keep'
        else current_action)
    if ((last_action is None or current_action is None)
            and current['end'] > last['end']):
        # Legacy no-action behavior: the earlier marker owns a partial
        # overlap. Action-aware callers use explicit precedence below.
        return last, [carve(current, last['end'], current['end'])]
    current_wins = (
        effective_last_action is None
        or effective_current_action is None
        or (priority.get(effective_current_action, 0)
            >= priority.get(effective_last_action, 0))
    )
    if not current_wins:
        if current['end'] <= last['end']:
            logger.info(
                f"Dropping {current.get('category')!r} span "
                f"{current['start']:.1f}s-{current['end']:.1f}s nested inside "
                f"higher-priority {last.get('category')!r} span "
                f"{last['start']:.1f}s-{last['end']:.1f}s"
            )
            return last, []
        return last, [carve(current, last['end'], current['end'])]

    if current['end'] <= last['end']:
        before = carve(last, last['start'], current['start'])
        after = carve(last, current['end'], last['end'])
        # Current keeps its own span, so its bookkeeping stays valid.
        current_copy = current.copy()
        clip_dai_core_spans(current_copy, current_copy['start'], current_copy['end'])
        new_last = before if before['start'] < before['end'] else None
        entries = [current_copy]
        if after['start'] < after['end']:
            entries.append(after)
        return new_last, entries

    shortened_last = carve(last, last['start'], current['start'])
    if shortened_last['end'] <= shortened_last['start']:
        shortened_last = None
    # Current keeps its own span, so its bookkeeping stays valid.
    return shortened_last, [current.copy()]




def deduplicate_window_ads(all_ads: list[dict], merge_threshold: float = 5.0,
                           action_map: dict[str, str] | None = None) -> list[dict]:
    """Deduplicate and merge ads detected across multiple windows.

    When the same ad spans two windows, both windows may detect it.
    This function merges overlapping detections.

    Args:
        all_ads: Combined list of ads from all windows
        merge_threshold: Seconds within which ads are considered overlapping
        action_map: Feed's resolved category->action map (see
            ``AdDetector._resolve_segment_action_map``). When given, gates
            the merge so two raw window detections whose categories resolve
            to different actions are never fused into one marker before
            categories are normalized. None treats every category as the
            same action, unchanged.

    Returns:
        Deduplicated list with overlapping ads merged
    """
    if not all_ads:
        return []

    # Sort by start time
    all_ads = sorted(all_ads, key=lambda x: x['start'])

    # Merge overlapping ads
    merged = [all_ads[0].copy()]

    for current in all_ads[1:]:
        last = merged[-1]

        # Check for overlap (ads within threshold seconds are considered overlapping)
        if current['start'] <= last['end'] + merge_threshold:
            last_action = (resolve_category_action(
                last.get('category'), action_map) if action_map else None)
            current_action = (resolve_category_action(
                current.get('category'), action_map) if action_map else None)
            same_action = (
                action_map is None
                or effective_resolved_action(last, action_map)
                == effective_resolved_action(current, action_map))
            if not same_action:
                new_last, new_entries = split_conflicting_action_span(
                    last, current, last_action, current_action)
                if new_last is None:
                    merged.pop()
                else:
                    merged[-1] = new_last
                merged.extend(new_entries)
                logger.debug(
                    f"Not merging {last.get('category')!r} and "
                    f"{current.get('category')!r} (different resolved "
                    f"actions) at window-dedup"
                )
                continue
            note_fold(last, current)
            # Merge: extend end time if current goes further
            if current['end'] > last['end']:
                last['end'] = current['end']
                if current.get('end_text'):
                    last['end_text'] = current['end_text']
            # Keep higher confidence
            if current.get('confidence', 0) > last.get('confidence', 0):
                last['confidence'] = current['confidence']
            # Keep sponsor and reason as a consistent pair from the SAME member
            # (mirrors _merge_detection_results): a merged marker must never show
            # one ad's sponsor with another ad's description. The longer reason is
            # the content-aware one, so take its sponsor with it.
            current_reason = current.get('reason', '')
            last_reason = last.get('reason', '')
            if len(current_reason) > len(last_reason):
                last['reason'] = current_reason
                last['sponsor'] = current.get('sponsor')
        else:
            merged.append(current.copy())

    if len(merged) < len(all_ads):
        logger.info(f"Window deduplication: {len(all_ads)} -> {len(merged)} ads")

    return merged


