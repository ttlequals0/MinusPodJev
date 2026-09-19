"""Live LLM pricing fetchers (OpenRouter API and the LiteLLM community JSON)
and the model-name normalizer they key on.

Vendored from MinusPod/src/config.py (normalize_model_key),
MinusPod/src/pricing_fetcher.py (fetch_openrouter_pricing, fetch_litellm_pricing,
_fetch_capped_body, LITELLM_PRICING_URL, PRICING_MAX_BYTES), and
MinusPod/src/utils/safe_http.py (read_response_capped, _declared_length,
ResponseTooLargeError, IncompleteResponseError).

These make live network calls at runtime, as in MinusPod. The MinusPod SSRF
layer (utils.safe_http.safe_get + URLTrust + utils.pinned_transport) is NOT
vendored: it guards arbitrary operator-typed URLs, whereas these two functions
fetch only the two fixed, operator-trusted HTTPS endpoints below. _fetch_capped_body
is reimplemented on a plain requests session that preserves the original timeout,
redirect cap, status check, and hard streamed-body byte cap. SSRFError is
therefore removed from the caught-exception tuples (it can no longer be raised).
"""
import json
import logging
import re
from typing import Protocol

import requests

logger = logging.getLogger(__name__)

# HTTP settings (config.py)
HTTP_MAX_REDIRECTS_API = 3            # LLM / PodcastIndex / webhook / pricing
HTTP_TIMEOUT_EXTERNAL = 15.0          # Third-party scraping (pricing sources)


def normalize_model_key(name: str) -> str:
    """Normalize a model name into a match key for pricing lookups.

    Examples:
        'Claude Sonnet 4.5'           -> 'claudesonnet45'
        'claude-sonnet-4-5-20250929'  -> 'claudesonnet45'
        'anthropic/claude-sonnet-4-5' -> 'claudesonnet45'
        'gpt-4o-mini'                 -> 'gpt4omini'
        'gpt-4o-2024-05-13'          -> 'gpt4o'

    Note: normalization is intentionally lossy (strips punctuation, hyphens).
    OpenRouter variants (:free, :extended) map to the same key as the base model.
    """
    # Strip provider prefix (anything before /)
    if '/' in name:
        name = name.split('/', 1)[1]
    # Strip OpenRouter variant suffixes (:free, :extended, :beta, :nitro, etc.)
    name = re.sub(r':[a-zA-Z]+$', '', name)
    # Strip date suffixes: YYYYMMDD or YYYY-MM-DD at end (2020-2039 range)
    name = re.sub(r'-?20[2-3]\d-?\d{2}-?\d{2}$', '', name)
    # Lowercase, remove everything non-alphanumeric
    return re.sub(r'[^a-z0-9]', '', name.lower())


# ========== Streaming byte-cap reader (utils/safe_http.py) ==========

class ResponseTooLargeError(Exception):
    """Raised when a streamed response exceeds the caller-supplied cap."""


class IncompleteResponseError(Exception):
    """Raised when a body ends before its declared Content-Length."""


class _ChunkedResponse(Protocol):
    headers: object

    def iter_content(self, chunk_size: int) -> object: ...


def read_response_capped(
    response: _ChunkedResponse, max_bytes: int, chunk_size: int = 65536
) -> bytes:
    """Stream a response body, raising above max_bytes or on a short read:
    a connection truncated mid-body ends iter_content without raising."""
    buf = bytearray()
    for chunk in response.iter_content(chunk_size=chunk_size):
        if not chunk:
            continue
        if len(buf) + len(chunk) > max_bytes:
            raise ResponseTooLargeError(
                f"response exceeds {max_bytes} bytes (had {len(buf)}, chunk {len(chunk)})"
            )
        buf.extend(chunk)

    expected = _declared_length(response)
    if expected is not None and len(buf) < expected:
        raise IncompleteResponseError(
            f"body ended at {len(buf)} of {expected} declared bytes"
        )
    return bytes(buf)


def _declared_length(response: _ChunkedResponse) -> int | None:
    """Declared Content-Length, or None when absent, malformed, or the body was
    content-encoded: iter_content decodes gzip, so the header counts encoded bytes."""
    headers = response.headers
    encoding = (headers.get('Content-Encoding') or '').strip().lower()
    if encoding and encoding != 'identity':
        return None
    raw = headers.get('Content-Length')
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None

# ========== Pricing fetchers (pricing_fetcher.py) ==========

# Hard byte cap on pricing response bodies. These are operator-trusted but
# remote hosts; without a cap a hostile or broken host could stream a
# multi-gigabyte body and OOM the worker. 16 MB leaves ample headroom for
# LiteLLM's full model-price JSON (a few MB and growing) while bounding memory.
PRICING_MAX_BYTES = 16 * 1024 * 1024


def _fetch_capped_body(url: str) -> bytes:
    """GET a pricing URL, enforcing the status check and a hard byte cap on the
    streamed body.

    Keeps ``raise_for_status`` so a non-2xx pricing response fails cleanly into
    the caller's ConnectionError fallback. Raises ``requests.RequestException``
    or ``ResponseTooLargeError``.
    """
    session = requests.Session()
    session.max_redirects = HTTP_MAX_REDIRECTS_API
    resp = session.get(url, timeout=HTTP_TIMEOUT_EXTERNAL, stream=True)
    try:
        resp.raise_for_status()
        return read_response_capped(resp, PRICING_MAX_BYTES)
    finally:
        resp.close()
        session.close()


def fetch_openrouter_pricing() -> list[dict]:
    """Fetch pricing from OpenRouter's /api/v1/models endpoint.

    Returns list of dicts:
      [{match_key, raw_model_id, display_name,
        input_cost_per_mtok, output_cost_per_mtok}, ...]
    """
    try:
        body = _fetch_capped_body('https://openrouter.ai/api/v1/models')
    except (requests.RequestException, ResponseTooLargeError,
            IncompleteResponseError) as exc:
        raise ConnectionError(f"Failed to fetch OpenRouter pricing: {exc}") from exc

    results = []
    for model in json.loads(body).get('data', []):
        pricing = model.get('pricing', {})
        try:
            input_per_mtok = float(pricing.get('prompt', '0')) * 1_000_000
            output_per_mtok = float(pricing.get('completion', '0')) * 1_000_000
        except (ValueError, TypeError):
            logger.debug(f"Skipping model with unparseable pricing: {model.get('id')}")
            continue

        raw_id = model.get('id', '')
        display_name = model.get('name', raw_id)
        key = normalize_model_key(raw_id)

        logger.debug(
            f"OpenRouter pricing: {raw_id} -> match_key={key} "
            f"in=${input_per_mtok:.4f}/Mtok out=${output_per_mtok:.4f}/Mtok"
        )

        results.append({
            'match_key': key,
            'raw_model_id': raw_id,
            'display_name': display_name,
            'input_cost_per_mtok': round(input_per_mtok, 4),
            'output_cost_per_mtok': round(output_per_mtok, 4),
        })

    return results


LITELLM_PRICING_URL = (
    'https://raw.githubusercontent.com/BerriAI/litellm/main/'
    'model_prices_and_context_window.json'
)


def fetch_litellm_pricing(provider_filter: str | None = None) -> list[dict]:
    """Fetch pricing from the LiteLLM community pricing JSON.

    This is a fallback source when the primary provider fetch returns nothing
    (e.g. pricepertoken page missing, unknown provider domain). The JSON is
    maintained by the LiteLLM project and updated frequently.

    Args:
        provider_filter: Optional ``litellm_provider`` value to filter on
            ('anthropic', 'openai', 'bedrock', etc.). None returns all providers
            that report per-token input/output costs.

    Returns list of dicts in the same shape as fetch_openrouter_pricing.
    """
    try:
        body = _fetch_capped_body(LITELLM_PRICING_URL)
    except (requests.RequestException, ResponseTooLargeError,
            IncompleteResponseError) as exc:
        raise ConnectionError(f"Failed to fetch LiteLLM pricing: {exc}") from exc

    try:
        raw = json.loads(body)
    except ValueError as exc:
        raise ConnectionError(f"LiteLLM pricing response was not valid JSON: {exc}") from exc

    results: list[dict] = []
    for raw_id, spec in raw.items():
        if raw_id == 'sample_spec' or not isinstance(spec, dict):
            continue
        try:
            input_cost = spec.get('input_cost_per_token')
            output_cost = spec.get('output_cost_per_token')
            if input_cost is None or output_cost is None:
                continue
            if provider_filter and spec.get('litellm_provider') != provider_filter:
                continue
            input_per_mtok = float(input_cost) * 1_000_000
            output_per_mtok = float(output_cost) * 1_000_000
        except (ValueError, TypeError):
            logger.debug(f"Skipping LiteLLM entry with unparseable pricing: {raw_id}")
            continue

        key = normalize_model_key(raw_id)
        results.append({
            'match_key': key,
            'raw_model_id': raw_id,
            'display_name': raw_id,
            'input_cost_per_mtok': round(input_per_mtok, 4),
            'output_cost_per_mtok': round(output_per_mtok, 4),
        })

    return results
