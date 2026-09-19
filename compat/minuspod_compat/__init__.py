"""minuspod_compat: a self-contained vendored copy of the MinusPod prompt,
parsing, schema, sponsor, window, and pricing helpers that the jev proxy and
the offline benchmark import.

Every symbol here is copied from MinusPod/src (see each submodule's provenance
header). The package depends only on the standard library plus ``requests``
(used by the pricing fetchers). It imports nothing from the MinusPod app.
"""
from .windows import create_windows
from .prompt import (
    format_window_prompt,
    get_static_system_prompt,
    strip_comments_from_prompt,
    scrub_description,
    utc_now_iso,
    format_time,
    USER_PROMPT_TEMPLATE,
    SEGMENT_ID_SYSTEM_SECTION,
    SEGMENT_ID_WINDOW_RULES,
)
from .parse import (
    parse_ads_from_response,
    parse_id_ads_from_response,
    resolve_segment_id_ads,
    extract_json_ads_array,
)
from .boundaries import deduplicate_window_ads
from .schema import (
    AD_DETECTION_JSON_SCHEMA,
    AD_DETECTION_MAX_TOKENS,
    SEGMENT_CATEGORIES,
    SPONSOR_PRIORITY_FIELDS,
    SPONSOR_ALIAS_FIELD_DESCRIPTION,
)
from .sponsors import DEFAULT_SYSTEM_PROMPT, SEED_SPONSORS, SPONSOR_ALIASES

# pricing pulls in `requests`; import it lazily so consumers that only need the
# prompt/schema/parse surface (the proxy) do not take that dependency.
_PRICING_EXPORTS = {
    "normalize_model_key",
    "fetch_litellm_pricing",
    "fetch_openrouter_pricing",
}


def __getattr__(name):
    if name in _PRICING_EXPORTS:
        from . import pricing
        return getattr(pricing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "create_windows",
    "format_window_prompt",
    "get_static_system_prompt",
    "strip_comments_from_prompt",
    "scrub_description",
    "utc_now_iso",
    "format_time",
    "USER_PROMPT_TEMPLATE",
    "SEGMENT_ID_SYSTEM_SECTION",
    "SEGMENT_ID_WINDOW_RULES",
    "parse_ads_from_response",
    "parse_id_ads_from_response",
    "resolve_segment_id_ads",
    "extract_json_ads_array",
    "deduplicate_window_ads",
    "AD_DETECTION_JSON_SCHEMA",
    "AD_DETECTION_MAX_TOKENS",
    "SEGMENT_CATEGORIES",
    "SPONSOR_PRIORITY_FIELDS",
    "SPONSOR_ALIAS_FIELD_DESCRIPTION",
    "DEFAULT_SYSTEM_PROMPT",
    "SEED_SPONSORS",
    "SPONSOR_ALIASES",
    "normalize_model_key",
    "fetch_litellm_pricing",
    "fetch_openrouter_pricing",
]
