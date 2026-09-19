"""Sponsor constants, gazetteer, ad-reason sponsor labeler, and the default
system prompt.

Vendored from MinusPod/src/utils/constants.py and
MinusPod/src/sponsor_service.py (SponsorService.extract_sponsor_from_reason,
de-classed into a module function). Behavior unchanged from the originals.
"""
import re

# Invalid sponsor values that indicate extraction failure or garbage data.
# Used by ad_detector (validate_ads_from_response, _extract_sponsor_from_reason)
# and text_pattern_matcher (create_pattern_from_ad).
INVALID_SPONSOR_VALUES = frozenset({
    'none', 'unknown', 'null', 'n/a', 'na', '', 'no', 'yes',
    'ad', 'ads', 'sponsor', 'sponsors', 'advertisement', 'advertisements',
    'multiple', 'various', 'detected', 'advertisement detected',
    'host read', 'host-read', 'mid-roll', 'pre-roll', 'post-roll',
    # Window-continuation notes the prompt itself asks for. 'note' is a
    # sponsor-candidate key, so a short one became the brand name and was
    # offered to pattern learning as a sponsor.
    'continues in next', 'continues from previous', 'continued',
    'continues', 'continuation',
    # Quantity words that open a reason ("Two consecutive cross-promotion
    # ads") are the first capitalized run, and became the sponsor. Only a
    # whole run is rejected, so "Five Guys" still labels. A stored registry
    # row of one bare count word also stops matching, which is intended.
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight',
    'nine', 'ten', 'several', 'both',
})

# or that contains an unambiguously meta substring. Real sponsor names never
# do. The text_pattern_matcher rejects these later, but catching them at
# parse time keeps junk out of the ad dict in the first place.
SPONSOR_REASONING_PREFIXES = (
    'inferred from', 'inferred', 'based on', 'according to',
    'likely ', 'possibly ', 'may be ', 'appears to ', 'seems to ',
    'detected as ', 'classified as ', 'regular discussion',
)
SPONSOR_REASONING_SUBSTRINGS = (
    ' in transcript', 'audio signal', 'no spoken content',
    'gap in transcript', 'volume anomaly',
)

# The substrings above only decide for text short enough to be nothing but a
# rationale; a full description can mention the transcript in passing. A
# rationale-shaped prefix still decides at any length.
SPONSOR_RATIONALE_SUBSTRING_MAX_CHARS = 200

SPONSOR_MAX_NAME_CHARS = 60

# Backstop on the detector's free-text reason. Generous on purpose: the old
# 300/150 caps put a literal "..." in the UI with nothing behind it (#591).
REASON_DESCRIPTION_MAX = 2000

_SQUASH_RE = re.compile(r'[^a-z0-9]')

# Longest a brand name is taken to be when no domain confirms where it ends;
# beyond this the model is describing rather than naming.
MAX_BRAND_WORDS = 4
# The labeler's span search is quadratic in a run's word count, so bound it.
# A brand a domain agrees with is never this long; past here it is prose.
MAX_SPAN_WORDS = 12


def squash_brand(text) -> str:
    """Brand text reduced to comparable characters, so a slug-style rendering
    still matches: "Jack Archer" and "jackarcher.com" both give 'jackarcher'."""
    return _SQUASH_RE.sub('', str(text).lower())


def is_sponsor_reasoning_rationale(text) -> bool:
    """True if `text` looks like an LLM reasoning sentence stored in a slot
    that should hold a brand name or ad description.

    Single source of truth for the 2.5.11 sponsor-field guard
    (ad_detector/prompts.py:_get_valid_sponsor_value), the 2.5.13 verification-miss
    `reason` filter (pattern_service.record_verification_misses), and the
    `_cleanup_low_mention_patterns` migration.
    """
    if not text:
        return False
    lowered = str(text).strip().lower()
    if lowered.startswith(SPONSOR_REASONING_PREFIXES):
        return True
    if len(lowered) <= SPONSOR_RATIONALE_SUBSTRING_MAX_CHARS:
        return any(s in lowered for s in SPONSOR_REASONING_SUBSTRINGS)
    return False


# Structural fields in LLM ad response objects that never contain sponsor info.
# Everything NOT in this set is a candidate for dynamic field scanning.
STRUCTURAL_FIELDS = frozenset({
    'start', 'end', 'start_time', 'end_time', 'start_timestamp', 'end_timestamp',
    'ad_start_timestamp', 'ad_end_timestamp', 'start_time_seconds', 'end_time_seconds',
    'confidence', 'end_text', 'is_ad', 'type', 'classification',
    'start_seconds', 'end_seconds', 'duration', 'duration_seconds',
    'music_bed', 'music_bed_confidence',
    # 'category' and its aliases: the sponsor scan falls back to any short
    # string field, so "self_promo" was being returned as the sponsor name.
    'category', 'segment_type',
})

# Keywords to match against any JSON key for fuzzy sponsor field detection.
SPONSOR_PATTERN_KEYWORDS = [
    'sponsor', 'brand', 'advertiser', 'company', 'product', 'ad_name', 'note'
]

# Invalid capture words - common English words that indicate regex captured garbage
# e.g., "not an advertisement" -> regex captures "not an" as sponsor.
# Distinct from NON_BRAND_WORDS below: this set targets English filler/
# grammatical words that appear at the START of a captured sponsor name
# (validate_extracted_sponsor in ad_detector). NON_BRAND_WORDS targets
# ad-domain vocabulary that follows or surrounds a sponsor mention.
INVALID_SPONSOR_CAPTURE_WORDS = frozenset({
    'not', 'no', 'this', 'that', 'the', 'a', 'an', 'another',
    'consistent', 'possible', 'potential', 'likely', 'seems',
    'is', 'was', 'are', 'were', 'with', 'from', 'for', 'by',
    'clear', 'any', 'some', 'host', 'their', 'its', 'our',
})

# Ad-domain vocabulary that appears in ad reasons / Claude output but is
# never a brand name. Used by ad_detector to filter spurious "sponsor"
# captures pulled from reason strings like "sponsor read" or "ad segment".
# This set is a strict superset of the inline excluded_words previously
# defined in extract_sponsor_names (the latter targeted the same domain
# but was narrower).
NON_BRAND_WORDS = frozenset({
    'ad', 'ads', 'sponsor', 'sponsored', 'advertisement', 'commercial',
    'host', 'read', 'segment', 'content', 'break', 'detected', 'detection',
    'network', 'inserted', 'dynamically', 'transition', 'promotional',
    'promo', 'promotion', 'mention', 'mentioned', 'plug', 'spot',
    'the', 'and', 'for', 'with', 'from', 'this', 'that', 'into',
    'brand', 'tagline', 'product', 'pitch', 'marketing', 'copy',
    'complete', 'partial', 'full', 'brief', 'short', 'long',
    'message', 'insert', 'mid', 'roll', 'pre', 'post',
})

# Vocabulary the model reaches for when describing an ad's shape or evidence,
# plus the pronouns it quotes ("We'll be right back"). Read only by the
# sponsor labeler. Kept out of NON_BRAND_WORDS because that set also filters
# boundary-relocation keywords, where losing "back" or "block" costs hits.
REASON_DESCRIPTION_WORDS = frozenset({
    'orphaned', 'contiguous', 'dai', 'url', 'back', 'block', 'lead',
    'fragment', 'leftover', 'confirmed', 'merged', 'missed', 'spots',
    'we', 'll', 'i', 'you', 'they', 'he', 'she', 'it', 'to',
})

NEGATION_WORDS = frozenset({
    'not', 'no', 'non', 'never', 'isnt', 'arent', 'wasnt', 'without',
})

# Words that only appear in a reason when the model is describing advertising.
AD_LANGUAGE_WORDS = frozenset({
    'ad', 'ads', 'advert', 'adverts', 'advertisement', 'advertisements',
    'advertiser', 'advertisers', 'advertising', 'sponsor', 'sponsors',
    'sponsored', 'sponsorship', 'commercial', 'commercials', 'promo',
    'promos', 'promotion', 'promotional', 'preroll', 'midroll', 'postroll',
    'dai', 'endorsement', 'infomercial', 'spot', 'spots',
})


def mentions_advertising(text) -> bool:
    """True if `text` calls the span an ad, the positive evidence the detection
    gate needs. Separate from the sponsor labeler, which answers what the
    advertiser is called and names the first capitalized word of any sentence.
    """
    if not text:
        return False
    words = re.findall(r'[a-z]+', str(text).lower())
    # A negated mention is the model saying the span is not an ad, so it is not
    # evidence that it is. Two tokens back covers "not a sponsor read".
    return any(w in AD_LANGUAGE_WORDS
               and NEGATION_WORDS.isdisjoint(words[max(0, i - 2):i])
               for i, w in enumerate(words))


# TLDs recognized in spoken "X dot com" transcript prose.
DOMAIN_TLDS = frozenset({'com', 'org', 'net', 'io', 'co'})

# TLDs a sponsor URL in an ad reason is written with. Wider than the spoken
# set: a written URL carries TLDs a host would not say aloud.
SPONSOR_DOMAIN_TLDS = DOMAIN_TLDS | frozenset({
    'tv', 'fm', 'us', 'app', 'shop', 'store', 'ai', 'edu',
})

# Classifications from LLM that indicate non-ad content
NOT_AD_CLASSIFICATIONS = frozenset({
    'content', 'not_ad', 'editorial', 'organic',
    'show_content', 'regular_content', 'interview',
    'conversation', 'segment', 'topic'
})


# ========== Sponsor labeler (from sponsor_service.py) ==========

# A brand is a capitalized run. Dot and slash are outside the character class,
# so "Patreon.com/Show" breaks into "Patreon" and "Show" rather than one run.
_BRAND_RUN_RE = re.compile(
    r"[A-Z][A-Za-z0-9&'\u2019-]*(?:\s+[A-Z][A-Za-z0-9&'\u2019-]*)*")
# Bounded quantifier: an unbounded run of [A-Za-z0-9-] here is the
# py/polynomial-redos shape fixed in 1.1.1. 63 is the DNS label limit.
_DOMAIN_RE = re.compile(
    r'\b([A-Za-z0-9][A-Za-z0-9-]{0,62})\.(?:%s)\b' % '|'.join(sorted(SPONSOR_DOMAIN_TLDS)),
    re.IGNORECASE)
_RUN_SPLIT_RE = re.compile(r"[\s'\u2019-]+")
_LABELER_STOPWORDS = NON_BRAND_WORDS | REASON_DESCRIPTION_WORDS
# Parts allowed inside a leading hyphenated descriptor ("Host-read") that is
# dropped whole; a single non-descriptor part (Full-Circle) keeps the token.
_HYPHEN_DESCRIPTOR_WORDS = INVALID_SPONSOR_CAPTURE_WORDS | NON_BRAND_WORDS


def _brand_run_words(run: str) -> list[str] | None:
    """Words of `run` with leading filler dropped, or None if it names nothing.

    Only INVALID_SPONSOR_CAPTURE_WORDS come off the front; trimming ad
    vocabulary here cost the first word of real names ("Full Circle").
    """
    words = run.split()
    while words:
        head = words[0].lower()
        parts = [p for p in head.split('-') if p]
        if head in INVALID_SPONSOR_CAPTURE_WORDS:
            words.pop(0)
        elif len(parts) > 1 and all(p in _HYPHEN_DESCRIPTOR_WORDS for p in parts):
            words.pop(0)
        else:
            break
    if not words:
        return None
    run = ' '.join(words)
    if len(run) < 3 or run.lower() in INVALID_SPONSOR_VALUES:
        return None
    parts = [p for p in _RUN_SPLIT_RE.split(run.lower()) if p]
    if all(p in _LABELER_STOPWORDS for p in parts):
        return None
    return None if is_sponsor_reasoning_rationale(run) else words


def _starts_any(domains):
    """Predicate: some domain begins with the given brand head."""
    return lambda head: any(d.startswith(head) for d in domains)


def extract_sponsor_from_reason(text: str) -> str | None:
    """Extract a sponsor name from an LLM ad-reason string, else None.

    A brand is a capitalized run narrowed to the span a domain in the same
    text agrees with; the model rewords the reason on every run, so a
    pattern keyed to a phrasing only covers the sample it was written for.
    """
    if not text:
        return None
    # A reason that is entirely the model explaining itself names no
    # advertiser, and a capitalized run inside it is just its first word.
    if is_sponsor_reasoning_rationale(text):
        return None
    # Prose that never mentions advertising names no advertiser either.
    # Without this the first capitalized word of any sentence becomes a
    # brand: "Discussion of the guest's new book" gave "Discussion".
    if not mentions_advertising(text):
        return None
    # Input cap, same reason as the bounded quantifiers above: this runs on
    # whatever string the model put in the field.
    text = text[:REASON_DESCRIPTION_MAX]

    # The first advertiser named labels the break, so stop at the first
    # usable run: a later one with a URL must not win the label.
    words = next(
        (w for w in (_brand_run_words(m.group(0))
                     for m in _BRAND_RUN_RE.finditer(text)) if w),
        None)
    if not words:
        return None

    domains = {squash_brand(m.group(1)) for m in _DOMAIN_RE.finditer(text)}
    # A domain names where the brand both starts and ends, so search spans
    # rather than prefixes: that narrows "Full ZipRecruiter" without a
    # blind leading trim. Leftmost and longest first, so "Jack Archer" is
    # not cut to "Jack" nor "Belmont Park" to "Park".
    span_words = words[:MAX_SPAN_WORDS]
    spans = [(squash_brand(' '.join(span_words[i:j])), i, j)
             for i in range(len(span_words))
             for j in range(len(span_words), i, -1)]
    for match_domain in (domains.__contains__, _starts_any(domains)):
        for head, i, j in spans:
            if head and match_domain(head):
                return ' '.join(span_words[i:j])
    # Nothing agrees, so there is no signal for where the brand ends. Cap
    # it: past this a run is the model describing the product, not naming
    # a brand ("LEGO Land Discovery Center Westchester Ninjago event").
    return ' '.join(words[:MAX_BRAND_WORDS])


# ========== Sponsor gazetteer (SEED_SPONSORS) ==========

SEED_SPONSORS = [
    {"name": "Athletic Greens", "aliases": ["AG1", "AG One"], "category": "health"},
    {"name": "BetterHelp", "aliases": ["Better Help"], "category": "health"},
    {"name": "Squarespace", "aliases": ["Square Space"], "category": "tech"},
    {"name": "Shopify", "aliases": [], "category": "tech"},
    {"name": "HelloFresh", "aliases": ["Hello Fresh"], "category": "food"},
    {"name": "NordVPN", "aliases": ["Nord VPN"], "category": "vpn"},
    {"name": "ExpressVPN", "aliases": ["Express VPN"], "category": "vpn"},
    {"name": "ZipRecruiter", "aliases": ["Zip Recruiter"], "category": "jobs"},
    {"name": "SimpliSafe", "aliases": ["Simpli Safe"], "category": "home"},
    {"name": "Mint Mobile", "aliases": ["MintMobile"], "category": "telecom"},
    {"name": "MasterClass", "aliases": ["Master Class"], "category": "education"},
    {"name": "Rocket Money", "aliases": ["RocketMoney", "Truebill"], "category": "finance"},
    {"name": "DoorDash", "aliases": ["Door Dash"], "category": "food"},
    {"name": "HubSpot", "aliases": ["Hub Spot"], "category": "tech"},
    {"name": "NetSuite", "aliases": ["Net Suite"], "category": "tech"},
    {"name": "Amazon", "aliases": [], "category": "retail"},
    {"name": "Audible", "aliases": [], "category": "entertainment"},
    {"name": "Factor", "aliases": [], "category": "food"},
    {"name": "Calm", "aliases": [], "category": "health"},
    {"name": "Headspace", "aliases": ["Head Space"], "category": "health"},
    {"name": "Indeed", "aliases": [], "category": "jobs"},
    {"name": "LinkedIn", "aliases": ["LinkedIn Jobs"], "category": "jobs"},
    {"name": "Stamps.com", "aliases": ["Stamps"], "category": "business"},
    {"name": "Ring", "aliases": [], "category": "home"},
    {"name": "ADT", "aliases": [], "category": "home"},
    {"name": "Casper", "aliases": [], "category": "home"},
    {"name": "Helix Sleep", "aliases": ["Helix"], "category": "home"},
    {"name": "Purple", "aliases": [], "category": "home"},
    {"name": "Brooklinen", "aliases": [], "category": "home"},
    {"name": "Bombas", "aliases": [], "category": "apparel"},
    {"name": "Manscaped", "aliases": [], "category": "personal"},
    {"name": "Dollar Shave Club", "aliases": ["DSC"], "category": "personal"},
    {"name": "Harry's", "aliases": ["Harrys"], "category": "personal"},
    {"name": "Quip", "aliases": [], "category": "personal"},
    {"name": "Hims", "aliases": [], "category": "health"},
    {"name": "Hers", "aliases": [], "category": "health"},
    {"name": "Roman", "aliases": [], "category": "health"},
    {"name": "Function of Beauty", "aliases": [], "category": "personal"},
    {"name": "Native", "aliases": [], "category": "personal"},
    {"name": "Liquid IV", "aliases": ["Liquid I.V."], "category": "health"},
    {"name": "Athletic Brewing", "aliases": [], "category": "beverage"},
    {"name": "Magic Spoon", "aliases": [], "category": "food"},
    {"name": "Thrive Market", "aliases": [], "category": "food"},
    {"name": "Butcher Box", "aliases": ["ButcherBox"], "category": "food"},
    {"name": "Blue Apron", "aliases": [], "category": "food"},
    {"name": "Uber Eats", "aliases": ["UberEats"], "category": "food"},
    {"name": "Grubhub", "aliases": ["Grub Hub"], "category": "food"},
    {"name": "Instacart", "aliases": [], "category": "food"},
    {"name": "Credit Karma", "aliases": [], "category": "finance"},
    {"name": "SoFi", "aliases": [], "category": "finance"},
    {"name": "Acorns", "aliases": [], "category": "finance"},
    {"name": "Betterment", "aliases": [], "category": "finance"},
    {"name": "Wealthfront", "aliases": [], "category": "finance"},
    {"name": "PolicyGenius", "aliases": ["Policy Genius"], "category": "finance"},
    {"name": "Lemonade", "aliases": [], "category": "finance"},
    {"name": "State Farm", "aliases": [], "category": "finance"},
    {"name": "Progressive", "aliases": [], "category": "finance"},
    {"name": "Geico", "aliases": [], "category": "finance"},
    {"name": "Liberty Mutual", "aliases": [], "category": "finance"},
    {"name": "T-Mobile", "aliases": ["TMobile"], "category": "telecom"},
    {"name": "Visible", "aliases": [], "category": "telecom"},
    {"name": "FanDuel", "aliases": ["Fan Duel"], "category": "gambling"},
    {"name": "DraftKings", "aliases": ["Draft Kings"], "category": "gambling"},
    {"name": "BetMGM", "aliases": ["Bet MGM"], "category": "gambling"},
    {"name": "Toyota", "aliases": [], "category": "auto"},
    {"name": "Hyundai", "aliases": [], "category": "auto"},
    {"name": "CarMax", "aliases": ["Car Max"], "category": "auto"},
    {"name": "Carvana", "aliases": [], "category": "auto"},
    {"name": "eBay Motors", "aliases": [], "category": "auto"},
    {"name": "ZocDoc", "aliases": ["Zoc Doc"], "category": "health"},
    {"name": "GoodRx", "aliases": ["Good Rx"], "category": "health"},
    {"name": "Care/of", "aliases": ["Care of", "Careof"], "category": "health"},
    {"name": "Ritual", "aliases": [], "category": "health"},
    {"name": "Seed", "aliases": [], "category": "health"},
    {"name": "Monday.com", "aliases": ["Monday"], "category": "tech"},
    {"name": "Notion", "aliases": [], "category": "tech"},
    {"name": "Canva", "aliases": [], "category": "tech"},
    {"name": "Grammarly", "aliases": [], "category": "tech"},
    {"name": "Babbel", "aliases": [], "category": "education"},
    {"name": "Rosetta Stone", "aliases": [], "category": "education"},
    {"name": "Blinkist", "aliases": [], "category": "education"},
    {"name": "Raycon", "aliases": [], "category": "electronics"},
    {"name": "Bose", "aliases": [], "category": "electronics"},
    {"name": "MacPaw", "aliases": ["CleanMyMac"], "category": "tech"},
    {"name": "Green Chef", "aliases": ["GreenChef"], "category": "food"},
    {"name": "Magic Mind", "aliases": [], "category": "beverage"},
    {"name": "Honeylove", "aliases": ["Honey Love"], "category": "apparel"},
    {"name": "Cozy Earth", "aliases": [], "category": "home"},
    {"name": "Quince", "aliases": [], "category": "apparel"},
    {"name": "LMNT", "aliases": ["Element"], "category": "health"},
    {"name": "Nutrafol", "aliases": [], "category": "health"},
    {"name": "Aura", "aliases": [], "category": "tech"},
    {"name": "OneSkin", "aliases": ["One Skin"], "category": "personal"},
    {"name": "Incogni", "aliases": [], "category": "tech"},
    {"name": "Gametime", "aliases": ["Game Time"], "category": "entertainment"},
    {"name": "1Password", "aliases": ["One Password"], "category": "tech"},
    {"name": "Bitwarden", "aliases": ["Bit Warden"], "category": "tech"},
    {"name": "CacheFly", "aliases": [], "category": "tech"},
    {"name": "Deel", "aliases": [], "category": "business"},
    {"name": "DeleteMe", "aliases": ["Delete Me"], "category": "tech"},
    {"name": "Framer", "aliases": [], "category": "tech"},
    {"name": "Miro", "aliases": [], "category": "tech"},
    {"name": "Monarch Money", "aliases": [], "category": "finance"},
    {"name": "OutSystems", "aliases": [], "category": "tech"},
    {"name": "Spaceship", "aliases": [], "category": "tech"},
    {"name": "Thinkst Canary", "aliases": [], "category": "tech"},
    {"name": "ThreatLocker", "aliases": [], "category": "tech"},
    {"name": "Vanta", "aliases": [], "category": "tech"},
    {"name": "Veeam", "aliases": [], "category": "tech"},
    {"name": "Zapier", "aliases": [], "category": "tech"},
    {"name": "Zscaler", "aliases": [], "category": "tech"},
    {"name": "Capital One", "aliases": [], "category": "finance"},
    {"name": "Ford", "aliases": [], "category": "auto"},
    {"name": "WhatsApp", "aliases": [], "category": "tech"},

    # 2.0.13 expansion: pb.json brands not previously in SEED (139 entries from Magellan AI / Podchaser / SponsorUnited)
    # automotive_transport
    {"name": "Lime", "aliases": [], "category": "automotive_transport"},
    {"name": "Lyft", "aliases": [], "category": "automotive_transport"},
    {"name": "Turo", "aliases": [], "category": "automotive_transport"},
    {"name": "Uber", "aliases": [], "category": "automotive_transport"},
    {"name": "Waymo", "aliases": [], "category": "automotive_transport"},

    # b2b_startup
    {"name": "Gusto", "aliases": [], "category": "b2b_startup"},
    {"name": "Meter", "aliases": [], "category": "b2b_startup"},
    {"name": "PagerDuty", "aliases": [], "category": "b2b_startup"},
    {"name": "Rippling", "aliases": [], "category": "b2b_startup"},
    {"name": "Splunk", "aliases": [], "category": "b2b_startup"},
    {"name": "Webflow", "aliases": [], "category": "b2b_startup"},

    # ecommerce_retail_dtc
    {"name": "Allbirds", "aliases": [], "category": "ecommerce_retail_dtc"},
    {"name": "Alo Yoga", "aliases": [], "category": "ecommerce_retail_dtc"},
    {"name": "Birchbox", "aliases": [], "category": "ecommerce_retail_dtc"},
    {"name": "Everlane", "aliases": [], "category": "ecommerce_retail_dtc"},
    {"name": "FabFitFun", "aliases": [], "category": "ecommerce_retail_dtc"},
    {"name": "GOAT", "aliases": [], "category": "ecommerce_retail_dtc"},
    {"name": "Gopuff", "aliases": [], "category": "ecommerce_retail_dtc"},
    {"name": "Lululemon", "aliases": [], "category": "ecommerce_retail_dtc"},
    {"name": "Outdoor Voices", "aliases": [], "category": "ecommerce_retail_dtc"},
    {"name": "Poshmark", "aliases": [], "category": "ecommerce_retail_dtc"},
    {"name": "Rothy's", "aliases": [], "category": "ecommerce_retail_dtc"},
    {"name": "Saatva", "aliases": [], "category": "ecommerce_retail_dtc"},
    {"name": "Shein", "aliases": [], "category": "ecommerce_retail_dtc"},
    {"name": "SKIMS", "aliases": [], "category": "ecommerce_retail_dtc"},
    {"name": "Stitch Fix", "aliases": [], "category": "ecommerce_retail_dtc"},
    {"name": "StockX", "aliases": [], "category": "ecommerce_retail_dtc"},
    {"name": "Temu", "aliases": [], "category": "ecommerce_retail_dtc"},
    {"name": "Ten Thousand", "aliases": [], "category": "ecommerce_retail_dtc"},
    {"name": "ThredUp", "aliases": [], "category": "ecommerce_retail_dtc"},
    {"name": "Vuori", "aliases": [], "category": "ecommerce_retail_dtc"},
    {"name": "Warby Parker", "aliases": [], "category": "ecommerce_retail_dtc"},
    {"name": "Wayfair", "aliases": [], "category": "ecommerce_retail_dtc"},

    # finance_fintech
    {"name": "Affirm", "aliases": [], "category": "finance_fintech"},
    {"name": "Bill.com", "aliases": [], "category": "finance_fintech"},
    {"name": "Brex", "aliases": [], "category": "finance_fintech"},
    {"name": "Chime", "aliases": [], "category": "finance_fintech"},
    {"name": "Coinbase", "aliases": [], "category": "finance_fintech"},
    {"name": "FreshBooks", "aliases": [], "category": "finance_fintech"},
    {"name": "Intuit", "aliases": [], "category": "finance_fintech"},
    {"name": "Klarna", "aliases": [], "category": "finance_fintech"},
    {"name": "Mercury", "aliases": [], "category": "finance_fintech"},
    {"name": "NerdWallet", "aliases": [], "category": "finance_fintech"},
    {"name": "Plaid", "aliases": [], "category": "finance_fintech"},
    {"name": "Public.com", "aliases": [], "category": "finance_fintech"},
    {"name": "QuickBooks", "aliases": [], "category": "finance_fintech"},
    {"name": "Ramp", "aliases": [], "category": "finance_fintech"},
    {"name": "Robinhood", "aliases": [], "category": "finance_fintech"},
    {"name": "Stripe", "aliases": [], "category": "finance_fintech"},
    {"name": "UnitedHealth Group", "aliases": [], "category": "finance_fintech"},
    {"name": "WebBank", "aliases": [], "category": "finance_fintech"},
    {"name": "Xero", "aliases": [], "category": "finance_fintech"},

    # food_beverage_nutrition
    {"name": "Alani Nu", "aliases": [], "category": "food_beverage_nutrition"},
    {"name": "Bloom Nutrition", "aliases": [], "category": "food_beverage_nutrition"},
    {"name": "EveryPlate", "aliases": [], "category": "food_beverage_nutrition"},
    {"name": "Huel", "aliases": [], "category": "food_beverage_nutrition"},
    {"name": "Imperfect Foods", "aliases": [], "category": "food_beverage_nutrition"},
    {"name": "McDonald's", "aliases": [], "category": "food_beverage_nutrition"},
    {"name": "OLIPOP", "aliases": [], "category": "food_beverage_nutrition"},
    {"name": "Poppi", "aliases": [], "category": "food_beverage_nutrition"},
    {"name": "Starbucks", "aliases": [], "category": "food_beverage_nutrition"},
    {"name": "Transparent Labs", "aliases": [], "category": "food_beverage_nutrition"},

    # gaming_sports_betting
    {"name": "Caesars Sportsbook", "aliases": [], "category": "gaming_sports_betting"},
    {"name": "ESPN Bet", "aliases": [], "category": "gaming_sports_betting"},
    {"name": "SeatGeek", "aliases": [], "category": "gaming_sports_betting"},
    {"name": "StubHub", "aliases": [], "category": "gaming_sports_betting"},

    # home_security
    {"name": "Pura", "aliases": [], "category": "home_security"},

    # insurance_legal
    {"name": "LegalZoom", "aliases": [], "category": "insurance_legal"},
    {"name": "Rocket Lawyer", "aliases": [], "category": "insurance_legal"},

    # media_streaming
    {"name": "Apple TV+", "aliases": [], "category": "media_streaming"},
    {"name": "Disney+", "aliases": [], "category": "media_streaming"},
    {"name": "HBO Max", "aliases": [], "category": "media_streaming"},
    {"name": "iHeartRadio", "aliases": [], "category": "media_streaming"},
    {"name": "Netflix", "aliases": [], "category": "media_streaming"},
    {"name": "Paramount+", "aliases": [], "category": "media_streaming"},
    {"name": "SiriusXM", "aliases": [], "category": "media_streaming"},
    {"name": "Spotify", "aliases": [], "category": "media_streaming"},
    {"name": "YouTube", "aliases": [], "category": "media_streaming"},
    {"name": "YouTube TV", "aliases": [], "category": "media_streaming"},

    # mental_health_wellness
    {"name": "Cerebral", "aliases": [], "category": "mental_health_wellness"},
    {"name": "Eight Sleep", "aliases": [], "category": "mental_health_wellness"},
    {"name": "Function Health", "aliases": [], "category": "mental_health_wellness"},
    {"name": "Inside Tracker", "aliases": [], "category": "mental_health_wellness"},
    {"name": "Joovv", "aliases": [], "category": "mental_health_wellness"},
    {"name": "Levels", "aliases": [], "category": "mental_health_wellness"},
    {"name": "Momentous", "aliases": [], "category": "mental_health_wellness"},
    {"name": "Noom", "aliases": [], "category": "mental_health_wellness"},
    {"name": "Ro", "aliases": [], "category": "mental_health_wellness"},
    {"name": "Talkspace", "aliases": [], "category": "mental_health_wellness"},
    {"name": "Thorne", "aliases": [], "category": "mental_health_wellness"},
    {"name": "WHOOP", "aliases": [], "category": "mental_health_wellness"},

    # tech_software_saas
    {"name": "Airtable", "aliases": [], "category": "tech_software_saas"},
    {"name": "Anthropic", "aliases": [], "category": "tech_software_saas"},
    {"name": "Asana", "aliases": [], "category": "tech_software_saas"},
    {"name": "Brilliant", "aliases": [], "category": "tech_software_saas"},
    {"name": "ClickUp", "aliases": [], "category": "tech_software_saas"},
    {"name": "Cloudflare", "aliases": [], "category": "tech_software_saas"},
    {"name": "CrowdStrike", "aliases": [], "category": "tech_software_saas"},
    {"name": "Cursor", "aliases": [], "category": "tech_software_saas"},
    {"name": "Databricks", "aliases": [], "category": "tech_software_saas"},
    {"name": "Datadog", "aliases": [], "category": "tech_software_saas"},
    {"name": "DocuSign", "aliases": [], "category": "tech_software_saas"},
    {"name": "Duolingo", "aliases": [], "category": "tech_software_saas"},
    {"name": "ElevenLabs", "aliases": [], "category": "tech_software_saas"},
    {"name": "Figma", "aliases": [], "category": "tech_software_saas"},
    {"name": "GitHub", "aliases": [], "category": "tech_software_saas"},
    {"name": "GitHub Copilot", "aliases": [], "category": "tech_software_saas"},
    {"name": "Klaviyo", "aliases": [], "category": "tech_software_saas"},
    {"name": "Linear", "aliases": [], "category": "tech_software_saas"},
    {"name": "Loom", "aliases": [], "category": "tech_software_saas"},
    {"name": "Mailchimp", "aliases": [], "category": "tech_software_saas"},
    {"name": "Midjourney", "aliases": [], "category": "tech_software_saas"},
    {"name": "Okta", "aliases": [], "category": "tech_software_saas"},
    {"name": "OpenAI", "aliases": [], "category": "tech_software_saas"},
    {"name": "Patreon", "aliases": [], "category": "tech_software_saas"},
    {"name": "Perplexity", "aliases": [], "category": "tech_software_saas"},
    {"name": "Retool", "aliases": [], "category": "tech_software_saas"},
    {"name": "Salesforce", "aliases": [], "category": "tech_software_saas"},
    {"name": "SendGrid", "aliases": [], "category": "tech_software_saas"},
    {"name": "ServiceNow", "aliases": [], "category": "tech_software_saas"},
    {"name": "Skillshare", "aliases": [], "category": "tech_software_saas"},
    {"name": "Slack", "aliases": [], "category": "tech_software_saas"},
    {"name": "Snowflake", "aliases": [], "category": "tech_software_saas"},
    {"name": "Substack", "aliases": [], "category": "tech_software_saas"},
    {"name": "Twilio", "aliases": [], "category": "tech_software_saas"},
    {"name": "Vercel", "aliases": [], "category": "tech_software_saas"},
    {"name": "Workday", "aliases": [], "category": "tech_software_saas"},
    {"name": "Zendesk", "aliases": [], "category": "tech_software_saas"},
    {"name": "Zoom", "aliases": [], "category": "tech_software_saas"},

    # telecom
    {"name": "AT&T", "aliases": [], "category": "telecom"},
    {"name": "Comcast", "aliases": [], "category": "telecom"},
    {"name": "Verizon", "aliases": [], "category": "telecom"},

    # travel_hospitality
    {"name": "Airbnb", "aliases": [], "category": "travel_hospitality"},
    {"name": "Booking.com", "aliases": [], "category": "travel_hospitality"},
    {"name": "Expedia", "aliases": [], "category": "travel_hospitality"},
    {"name": "Hopper", "aliases": [], "category": "travel_hospitality"},
    {"name": "Kayak", "aliases": [], "category": "travel_hospitality"},
    {"name": "Skyscanner", "aliases": [], "category": "travel_hospitality"},
    {"name": "Vrbo", "aliases": [], "category": "travel_hospitality"},
    {"name": "Zyn", "aliases": ["ZYN", "Zinn"], "category": "tobacco_nicotine"},
]


# ========== Default ad-detection system prompt ==========

DEFAULT_SYSTEM_PROMPT = """Analyze this podcast transcript and identify ALL advertisement segments.

DETECTION RULES:
- Host-read sponsor segments ARE ads. Any product promotion for compensation is an ad.
- An ad MUST contain promotional language in the transcript. You must be able to point to specific words (sponsor names, URLs, promo codes, product pitches, calls to action) that make it an ad.
- Include the transition phrase ("let's take a break") in the ad segment, not just the pitch.
- Ad breaks typically last 60-120 seconds. Shorter segments may indicate incomplete detection.
- If no ads are found in this window, return: []

WHAT IS NOT AN AD:
- Silence, pauses, or dead air between segments -- these are normal production gaps, not ads
- Topic transitions or content gaps where the host changes subjects
- Audio signal changes (volume shifts, tone changes) without any promotional transcript content
- A guest discussing their own work, book, or project in the context of the interview
- The host organically mentioning their own other shows, social media, or Patreon as part of conversation
- Brand names mentioned in passing as part of genuine topic discussion

PLATFORM-INSERTED ADS (these ARE ads -- flag them):
- Hosting platform pre/post-rolls: "Acast powers the world's best podcasts", "Hosted on Acast",
  "Spotify for Podcasters", "iHeart Radio", etc. These are promotional insertions by the hosting
  platform, not part of the show content. They typically bookend the episode.
- Cross-promotions for other podcasts: Segments promoting a different show (different host, different
  topic) inserted by the platform or network. These are ads even without promo codes.
- Network promos: Short produced segments advertising other shows on the same network.
- The distinction: if the HOST organically says "check out my other show" during conversation,
  that's not an ad. If a PRODUCED SEGMENT with different audio/voice promotes another show or
  the hosting platform itself, that IS an ad.

WHAT TO LOOK FOR:
- Transitions: "This episode is brought to you by...", "A word from our sponsors", "Let's take a break"
- Promo codes, vanity URLs (example.com/podcast), calls to action
- Product endorsements, sponsored content, promotional messages
- Network-inserted retail ads (may sound like radio commercials)
- Dynamically inserted ads that may differ in tone or cadence from the host content
- Short brand tagline ads (15-45 seconds): Network-inserted spots that sound like polished
  radio/TV commercials rather than host reads. They use concentrated marketing language
  ("bringing you the latest", "where innovation lands first", "explore what's new", "level up
  your game") without promo codes or URLs. They are typically voiced by someone other than the
  host and feel tonally distinct from the surrounding editorial content. Common structure: brand
  name + tagline + product category pitch + brand name repeat. Flag these even though they lack
  traditional ad markers like promo codes.

AUDIO SIGNALS:
Audio analysis may detect volume anomalies, DAI transitions, silence gaps, or labelled audio cues
(show stingers / break jingles known to bracket ad breaks on this show).
These signals are SUPPORTING EVIDENCE ONLY. They help locate potential ad boundaries but do NOT
constitute ads by themselves. You MUST find promotional content in the transcript (sponsor names,
URLs, promo codes, product pitches, calls to action) to flag a segment as an ad. A volume change
or silence gap with no promotional language is just normal audio production -- not an ad.
Unlabelled generic cues are weaker evidence than labelled template cues; the AUDIO SIGNALS block
states each cue's weight.

LABELLED AUDIO CUES: when the AUDIO SIGNALS list a labelled cue, treat it as a strong boundary
marker for the side of the ad break it sits on; the detailed handling (multi-cue breaks, where to
start and end the span) is supplied alongside the cue in the AUDIO SIGNALS block. The cue is never
an ad on its own.

COMMON PODCAST SPONSORS (high confidence if mentioned):
BetterHelp, Athletic Greens, AG1, Shopify, Amazon, Audible, Squarespace, HelloFresh, Factor, NordVPN, ExpressVPN, Mint Mobile, MasterClass, Calm, Headspace, ZipRecruiter, Indeed, LinkedIn Jobs, LinkedIn, Stamps.com, SimpliSafe, Ring, ADT, Casper, Helix Sleep, Purple, Brooklinen, Bombas, Manscaped, Dollar Shave Club, Harry's, Quip, Hims, Hers, Roman, Function of Beauty, Native, Liquid IV, Athletic Brewing, Magic Spoon, Thrive Market, Butcher Box, Blue Apron, DoorDash, Uber Eats, Grubhub, Instacart, Rocket Money, Credit Karma, SoFi, Acorns, Betterment, Wealthfront, PolicyGenius, Lemonade, State Farm, Progressive, Geico, Liberty Mutual, T-Mobile, Visible, FanDuel, DraftKings, BetMGM, Toyota, Hyundai, CarMax, Carvana, eBay Motors, ZocDoc, GoodRx, Care/of, Ritual, Seed, HubSpot, NetSuite, Monday.com, Notion, Canva, Grammarly, Babbel, Rosetta Stone, Blinkist, Raycon, Bose, MacPaw, CleanMyMac, Green Chef, Magic Mind, Honeylove, Cozy Earth, Quince, LMNT, Nutrafol, Aura, OneSkin, Incogni, Gametime, 1Password, Bitwarden, CacheFly, Deel, DeleteMe, Framer, Miro, Monarch Money, OutSystems, Spaceship, Thinkst Canary, ThreatLocker, Vanta, Veeam, Zapier, Zscaler, Capital One, Ford, WhatsApp

RETAIL/CONSUMER BRANDS (network-inserted ads):
Nordstrom, Macy's, Target, Walmart, Kohl's, Bloomingdale's, JCPenney, TJ Maxx, Home Depot, Lowe's, Best Buy, Costco, Gap, Old Navy, H&M, Zara, Nike, Adidas, Lululemon, Coach, Kate Spade, Michael Kors, Sephora, Ulta, Bath & Body Works, CVS, Walgreens, AutoZone, O'Reilly Auto Parts, Jiffy Lube, Midas, Gold Belly, Farmer's Dog, Caldera Lab, Monster Energy, Red Bull, Whole Foods, Trader Joe's, Kroger, GNC

AD BOUNDARY RULES:
- AD START: Include transition phrases like "Let's take a break", "A word from our sponsors"
- AD END: The ad ends when SHOW CONTENT resumes, NOT when the pitch ends. Wait for:
  - Topic change back to episode content
  - Host says "anyway", "alright", "so" and changes subject
  - AFTER the final URL mention (they often repeat it)
- MERGING: Multiple ads with gaps < 15 seconds = ONE segment

WINDOW CONTEXT:
This transcript may be a segment of a longer episode.
- If an ad appears to START before this segment, mark start as the first timestamp
- If an ad appears to CONTINUE past this segment, mark end as the last timestamp
- Note partial ads in the reason field

TIMESTAMP PRECISION:
Use the exact START timestamp from the [Xs] marker of the first ad segment.
Use the exact END timestamp from the [Xs] marker of the last ad segment.
Do not interpolate or estimate times between segments.

OUTPUT FORMAT:
Return ONLY a valid JSON array. No explanation, no markdown.

Each ad segment: {{"start": FLOAT_SECONDS, "end": FLOAT_SECONDS, "confidence": FLOAT_0_TO_1, "category": "sponsor|cross_promo|self_promo|interaction", "reason": "brief description", "end_text": "last 3-5 words"}}

"category" is REQUIRED on every ad object, with no exceptions. A response where any object omits "category" is invalid, even if you are confident the category is obvious from the reason text. Always write the key. See CATEGORY below for the exact allowed values.

ALL values for "start", "end", and "confidence" MUST be numeric (float). Never use strings like "high", "low", "medium", or percentages like "95%". Examples: "start": 45.0, "end": 82.0, "confidence": 0.95

CATEGORY:
Every ad object MUST also include "category", set to exactly one of:
- sponsor: a paid host read, a produced ad spot, a dynamically inserted ad (DAI), or a platform-inserted ad (hosting platform pre/post-rolls, etc.)
- cross_promo: a produced segment promoting a different show, inserted by the platform or network. A paid read promoting another podcast or show is sponsor, not cross_promo; use cross_promo only for unpaid promotion of shows from the same network or host.
- self_promo: a produced or inserted segment where the show promotes its own other content (another show, Patreon, merch, mailing list)
- interaction: a produced or inserted segment asking listeners to subscribe, rate, review, or follow the show
Three more categories exist (intro, outro, recap), but use them only when this prompt also contains a SHOW SEGMENTS section below. Without that section, always pick one of the four categories above.

EXAMPLE:
[45.0s - 48.0s] That's a great point. Let's take a quick break.
[48.5s - 52.0s] This episode is brought to you by Athletic Greens.
[52.5s - 78.0s] AG1 is the daily foundational nutrition supplement... Go to athleticgreens.com/podcast.
[78.5s - 82.0s] That's athleticgreens.com/podcast.
[82.5s - 86.0s] Now, back to our conversation.

Output: [{{"start": 45.0, "end": 82.0, "confidence": 0.98, "category": "sponsor", "reason": "Athletic Greens sponsor read", "end_text": "athleticgreens.com/podcast"}}]

NOT AN AD EXAMPLE (silence/content gap):
[290.0s - 293.0s] So that's really the core of what GPT-4 can do.
[293.5s - 296.0s] [silence]
[296.5s - 300.0s] Now the other thing I wanted to talk about is the fine-tuning process.

Output: []

SHORT BRAND TAGLINE EXAMPLE (this IS an ad):
[874.2s - 877.0s] FreshField Market, your destination for what's next in nutrition.
[877.0s - 886.0s] Curated by experts who know what works, we bring you the best in health and wellness.
[886.0s - 893.0s] Whether you're training hard, living well, or chasing your best self,
[893.0s - 898.5s] FreshField Market is where the future of wellness begins. Explore more at FreshField.

Output: [{{"start": 874.2, "end": 898.5, "confidence": 0.95, "category": "sponsor", "reason": "FreshField Market network-inserted brand tagline ad", "end_text": "wellness begins. Explore more at FreshField"}}]

Note: No promo code, no call to action -- but this is concentrated marketing copy
for a brand with product positioning language. It is not editorial content.

CROSS-PROMO EXAMPLE (this IS an ad, and its category is NOT sponsor):
[512.0s - 514.5s] Before we get back to it, a quick note.
[514.5s - 528.0s] Hey, it's Jamie from Tech Weekly. If you like this show, check out our
sister podcast Startup Stories for interviews with founders every Tuesday.
[528.0s - 531.0s] Now, back to today's episode.

Output: [{{"start": 512.0, "end": 531.0, "confidence": 0.9, "category": "cross_promo", "reason": "Produced cross-promotion for the sister podcast Startup Stories", "end_text": "back to today's episode"}}]

Note: a different voice promoting a different show, inserted by the platform or network.
Not a sponsor read, so "category" is "cross_promo", not "sponsor".{sponsor_database}"""


# ========== Sponsor name aliases (utils/constants.py) ==========


# Sponsor name aliases for common Whisper mishearings / spelling variants.
# Lookup is lowercase. The value is the canonical sponsor name stored on
# created patterns. Applied in ad_detector.learn_from_detections and
# pattern_service.record_verification_misses before sponsor-based gating so
# the variants merge into one pattern family instead of splitting across
# parallel misspelled entries.
SPONSOR_ALIASES = {
    # Xero
    'zero': 'Xero',
    'xerox': 'Xero',
    # 1Password
    '1 password': '1Password',
    'one password': '1Password',
    'one-password': '1Password',
    # Affirm
    'a firm': 'Affirm',
    # AG1 / Athletic Greens (SEED canonical is "Athletic Greens"; AG1 is an alias)
    'ag one': 'Athletic Greens',
    'ag 1': 'Athletic Greens',
    'a g one': 'Athletic Greens',
    'ag1': 'Athletic Greens',
    'athletic greens one': 'Athletic Greens',
    'athleticgreens': 'Athletic Greens',
    # Athlean-X
    'athlean x': 'Athlean-X',
    'athlean-x': 'Athlean-X',
    # BetMGM
    'bet mgm': 'BetMGM',
    'bet-mgm': 'BetMGM',
    # BetterHelp
    'better help': 'BetterHelp',
    'better-help': 'BetterHelp',
    # Birchbox
    'birch box': 'Birchbox',
    'birch-box': 'Birchbox',
    # Bitwarden
    'bit warden': 'Bitwarden',
    'bit-warden': 'Bitwarden',
    # Blue Apron
    'blueapron': 'Blue Apron',
    # Brex (skip 'brexit' - distinct noun)
    'brecks': 'Brex',
    # Butcher Box (SEED canonical is two-word form)
    'butcher box': 'Butcher Box',
    'butcher-box': 'Butcher Box',
    'butcherbox': 'Butcher Box',
    # CarMax
    'car max': 'CarMax',
    'car-max': 'CarMax',
    # Cloudflare
    'cloud flare': 'Cloudflare',
    'cloud-flare': 'Cloudflare',
    # Credit Karma
    'creditkarma': 'Credit Karma',
    # DeleteMe
    'delete me': 'DeleteMe',
    'delete-me': 'DeleteMe',
    # Dollar Shave Club
    'dollarshaveclub': 'Dollar Shave Club',
    # DoorDash
    'door dash': 'DoorDash',
    'door-dash': 'DoorDash',
    # DraftKings
    'draft kings': 'DraftKings',
    'draft-kings': 'DraftKings',
    # Eight Sleep
    'eight-sleep': 'Eight Sleep',
    '8 sleep': 'Eight Sleep',
    '8-sleep': 'Eight Sleep',
    'eightsleep': 'Eight Sleep',
    # EveryPlate
    'every plate': 'EveryPlate',
    'every-plate': 'EveryPlate',
    # ExpressVPN
    'express vpn': 'ExpressVPN',
    'express-vpn': 'ExpressVPN',
    # FabFitFun
    'fab fit fun': 'FabFitFun',
    'fab-fit-fun': 'FabFitFun',
    # FanDuel
    'fan duel': 'FanDuel',
    'fan-duel': 'FanDuel',
    # Gametime (SEED canonical)
    'game time': 'Gametime',
    'game-time': 'Gametime',
    'gametime': 'Gametime',
    # GitHub Copilot
    'co pilot': 'GitHub Copilot',
    'co-pilot': 'GitHub Copilot',
    'copilot': 'GitHub Copilot',
    'github-copilot': 'GitHub Copilot',
    # Gopuff
    'go puff': 'Gopuff',
    'go-puff': 'Gopuff',
    # GoodRx
    'good rx': 'GoodRx',
    'good-rx': 'GoodRx',
    # Green Chef
    'green chef': 'Green Chef',
    'green-chef': 'Green Chef',
    'greenchef': 'Green Chef',
    # Grubhub
    'grub hub': 'Grubhub',
    'grub-hub': 'Grubhub',
    # Harry's
    'harrys': "Harry's",
    # Headspace
    'head space': 'Headspace',
    'head-space': 'Headspace',
    # HelloFresh
    'hello fresh': 'HelloFresh',
    'hello-fresh': 'HelloFresh',
    # Hims / Hims & Hers
    "him's": 'Hims',
    'hims and hers': 'Hims & Hers',
    'hims & hers': 'Hims & Hers',
    # Honeylove (SEED canonical)
    'honey love': 'Honeylove',
    'honey-love': 'Honeylove',
    'honeylove': 'Honeylove',
    # HubSpot
    'hub spot': 'HubSpot',
    'hub-spot': 'HubSpot',
    'hubs pot': 'HubSpot',
    # Imperfect Foods
    'imperfect foods': 'Imperfect Foods',
    'imperfectfoods': 'Imperfect Foods',
    # Instacart
    'insta cart': 'Instacart',
    'insta-cart': 'Instacart',
    # LegalZoom
    'legal zoom': 'LegalZoom',
    'legal-zoom': 'LegalZoom',
    'legalzoom': 'LegalZoom',
    # Liquid IV (SEED canonical; "Liquid I.V." is the alias form)
    'liquid iv': 'Liquid IV',
    'liquid i v': 'Liquid IV',
    'liquid i.v.': 'Liquid IV',
    'liquidiv': 'Liquid IV',
    # LMNT (canonical matches existing SEED entry)
    'l m n t': 'LMNT',
    'element': 'LMNT',
    # Magic Mind
    'magic mind': 'Magic Mind',
    'magicmind': 'Magic Mind',
    # Magic Spoon
    'magic spoon': 'Magic Spoon',
    'magicspoon': 'Magic Spoon',
    # MasterClass
    'master class': 'MasterClass',
    'master-class': 'MasterClass',
    # Mercury
    'mercury bank': 'Mercury',
    'mercury-bank': 'Mercury',
    # Mint Mobile
    'mint mobile': 'Mint Mobile',
    'mint-mobile': 'Mint Mobile',
    'mintmobile': 'Mint Mobile',
    # Miro (skip 'mirror' - common word)
    'my ro': 'Miro',
    # Monarch Money
    'monarch money': 'Monarch Money',
    'monarch-money': 'Monarch Money',
    'monarchmoney': 'Monarch Money',
    # Myprotein
    'my protein': 'Myprotein',
    'myprotein': 'Myprotein',
    # NetSuite
    'net suite': 'NetSuite',
    'net-suite': 'NetSuite',
    # NordVPN
    'nord vpn': 'NordVPN',
    'nord-vpn': 'NordVPN',
    # OneSkin
    'one skin': 'OneSkin',
    'one-skin': 'OneSkin',
    # P90X
    'p ninety x': 'P90X',
    # Patreon
    'pay tree on': 'Patreon',
    'patron': 'Patreon',
    # Perplexity
    'perplexity ai': 'Perplexity',
    'perplexity-ai': 'Perplexity',
    # PolicyGenius
    'policy genius': 'PolicyGenius',
    'policy-genius': 'PolicyGenius',
    # Pura
    'pyura': 'Pura',
    # Raycon
    'ray con': 'Raycon',
    'ray-con': 'Raycon',
    # Retool
    're tool': 'Retool',
    # Rocket Lawyer / Money / Mortgage
    'rocketlawyer': 'Rocket Lawyer',
    'rocket money': 'Rocket Money',
    'rocket-money': 'Rocket Money',
    'rocketmoney': 'Rocket Money',
    'rocketmortgage': 'Rocket Mortgage',
    # Rogaine
    'ro gain': 'Rogaine',
    'ro-gaine': 'Rogaine',
    # SeatGeek
    'seat geek': 'SeatGeek',
    'seat-geek': 'SeatGeek',
    # Shopify
    'shop ify': 'Shopify',
    'shop a fly': 'Shopify',
    'shop fly': 'Shopify',
    # SimpliSafe
    'simpli safe': 'SimpliSafe',
    'simpli-safe': 'SimpliSafe',
    'simply safe': 'SimpliSafe',
    # Skyscanner
    'sky scanner': 'Skyscanner',
    'sky-scanner': 'Skyscanner',
    # SoFi (skip 'Sophie' - common name)
    'so fi': 'SoFi',
    'so-fi': 'SoFi',
    # Squarespace
    'square space': 'Squarespace',
    'square-space': 'Squarespace',
    # Stamps.com
    'stamp dot com': 'Stamps.com',
    # Stitch Fix
    'stitch fix': 'Stitch Fix',
    'stitch-fix': 'Stitch Fix',
    'stitchfix': 'Stitch Fix',
    # StubHub
    'stub hub': 'StubHub',
    'stub-hub': 'StubHub',
    # Substack
    'sub stack': 'Substack',
    'sub-stack': 'Substack',
    # Thrive Market
    'thrive market': 'Thrive Market',
    'thrivemarket': 'Thrive Market',
    # Transparent Labs
    'transparent labs': 'Transparent Labs',
    'transparentlabs': 'Transparent Labs',
    # Uber Eats
    'uber eats': 'Uber Eats',
    'uber-eats': 'Uber Eats',
    'ubereats': 'Uber Eats',
    # Vercel
    'ver sel': 'Vercel',
    'ver cell': 'Vercel',
    # Wealthfront
    'wealth front': 'Wealthfront',
    'wealth-front': 'Wealthfront',
    # Whoop
    'woop': 'Whoop',
    # ZipRecruiter
    'zip recruiter': 'ZipRecruiter',
    'zip-recruiter': 'ZipRecruiter',
    # ZocDoc
    'zoc doc': 'ZocDoc',
    'zoc-doc': 'ZocDoc',
    'zock doc': 'ZocDoc',
}
