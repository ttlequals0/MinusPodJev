"""Redact secrets from any string before it reaches a log."""

from __future__ import annotations

import re

_BEARER = re.compile(r"(?i)(bearer\s+)\S+")
_PASSWORD = re.compile(r'(?i)("?password"?\s*[:=]\s*)("[^"]*"|\S+)')
_URL_CREDENTIALS = re.compile(r"(?i)(https?://)[^/@\s]+@")


def redact(text: str) -> str:
    """Mask `Bearer <token>` and any `password` value; leave everything else."""
    text = _BEARER.sub(r"\1[REDACTED]", text)
    text = _PASSWORD.sub(r"\1[REDACTED]", text)
    text = _URL_CREDENTIALS.sub(r"\1[REDACTED]@", text)
    return text
