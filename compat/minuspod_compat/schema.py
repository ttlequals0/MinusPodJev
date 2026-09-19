"""Segment categories, sponsor field constants, and the ad-detection JSON schema.

Vendored from MinusPod/src/config.py (SEGMENT_CATEGORIES,
SEGMENT_CATEGORY_ALIASES, repair_segment_category, AD_DETECTION_MAX_TOKENS),
MinusPod/src/utils/constants.py (SPONSOR_PRIORITY_FIELDS), and
MinusPod/src/ad_detector/prompts.py (SPONSOR_ALIAS_FIELD_DESCRIPTION,
AD_DETECTION_JSON_SCHEMA). Behavior unchanged from the originals.
"""
import os

# Detection token budget (config.py:1411): env-aware, same default as MinusPod.
AD_DETECTION_MAX_TOKENS = int(os.environ.get('AD_DETECTION_MAX_TOKENS', '4096'))


# Segment categories (issue #565): what kind of content a marker spans. A
# marker may carry none: unset means no stage classified it, and only action
# resolution defaults (see normalize_segment_category).
SEGMENT_CATEGORIES = ('sponsor', 'cross_promo', 'self_promo', 'interaction',
                      'intro', 'outro', 'recap')


# Spelled-out forms a model reaches for instead of the canonical category.
SEGMENT_CATEGORY_ALIASES = {
    'self_promotion': 'self_promo',
    'selfpromo': 'self_promo',
    'cross_promotion': 'cross_promo',
    'crosspromo': 'cross_promo',
    'sponsorship': 'sponsor',
}


def repair_segment_category(value):
    """A known category from any spelling, or None. Spacing, case and hyphens
    vary between providers ("Cross-Promo"); the vocabulary does not. A position
    word like "pre-roll", or a bare "ad", is not a category and stays rejected.
    """
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower().replace('-', '_').replace(' ', '_')
    candidate = SEGMENT_CATEGORY_ALIASES.get(candidate, candidate)
    return candidate if candidate in SEGMENT_CATEGORIES else None


# Ordered list of field names to check for sponsor/advertiser name (priority order).
SPONSOR_PRIORITY_FIELDS = [
    'sponsor_name', 'advertiser', 'sponsor', 'brand', 'company', 'product', 'name'
]


# Shared by every sponsor alias in the detection schema. The aliases exist so
# a key-stripping backend cannot discard whichever one a model volunteers.
SPONSOR_ALIAS_FIELD_DESCRIPTION = (
    "The advertiser being promoted, when the segment names one. "
    "Fill at most one of the sponsor fields; omit them all otherwise."
)

AD_DETECTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "ads": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "start_id": {"type": "integer"},
                    "end_id": {"type": "integer"},
                    # The prompt requires end_text on every segment and the
                    # sponsor extractors read these names; a schema-enforcing
                    # decoder would silently strip anything absent here.
                    "end_text": {"type": "string"},
                    # Same enum as the repair schema above: an enforcing
                    # decoder cannot emit a synonym the repair map translates.
                    "category": {"type": "string", "enum": list(SEGMENT_CATEGORIES)},
                    "confidence": {"type": "number"},
                    "reason": {"type": "string"},
                    "note": {"type": "string"},
                    # Described on the extractor's first-choice field only:
                    # the model follows the schema here, so repeating it on all
                    # seven adds tokens and invites the multi-fill it warns off.
                    **{name: ({"type": "string",
                               "description": SPONSOR_ALIAS_FIELD_DESCRIPTION}
                              if name == SPONSOR_PRIORITY_FIELDS[0]
                              else {"type": "string"})
                       for name in SPONSOR_PRIORITY_FIELDS},
                },
            },
        },
    },
    "required": ["ads"],
}
