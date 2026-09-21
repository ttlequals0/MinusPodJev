"""Persistent runtime threshold settings."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RuntimeSettingsError(RuntimeError):
    """The runtime settings file cannot be read or written safely."""


class RuntimeSettingsValidationError(RuntimeSettingsError):
    """A requested runtime settings value is invalid."""


@dataclass(frozen=True)
class Thresholds:
    """Effective thresholds for one request."""

    detection_enter: float
    detection_stay: float
    review_evidence: float
    review_choice: float


def defaults_from_settings(config: Any) -> Thresholds:
    """Build startup defaults without retaining a mutable settings reference."""
    evidence = config.JEV_REVIEW_EVIDENCE_THRESHOLD
    choice = config.JEV_REVIEW_CHOICE_THRESHOLD
    return Thresholds(
        detection_enter=float(config.JEV_ENTER),
        detection_stay=float(config.JEV_STAY),
        review_evidence=float(config.JEV_ENTER if evidence is None else evidence),
        review_choice=float(config.JEV_ENTER if choice is None else choice),
    )


def _validate_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeSettingsValidationError(f"{name} must be a number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise RuntimeSettingsValidationError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise RuntimeSettingsValidationError(f"{name} must be between 0 and 1")
    return number


def validate_editable(values: dict[str, object]) -> Thresholds:
    """Validate the complete set of runtime-editable thresholds."""
    expected = {"detection_enter", "detection_stay", "review_evidence", "review_choice"}
    if set(values) != expected:
        raise RuntimeSettingsValidationError("thresholds must contain exactly four numeric fields")
    detection_enter = _validate_number(values["detection_enter"], "detection_enter")
    detection_stay = _validate_number(values["detection_stay"], "detection_stay")
    review_evidence = _validate_number(values["review_evidence"], "review_evidence")
    review_choice = _validate_number(values["review_choice"], "review_choice")
    if detection_enter < detection_stay:
        raise RuntimeSettingsValidationError("detection_enter must be at least detection_stay")
    return Thresholds(detection_enter, detection_stay, review_evidence, review_choice)


def _validate_persisted(values: dict[str, object], defaults: Thresholds) -> Thresholds:
    """Accept the former three-field file format with the startup stay default."""
    if set(values) == {"detection_enter", "review_evidence", "review_choice"}:
        values = {**values, "detection_stay": defaults.detection_stay}
    return validate_editable(values)


def read(path: str, defaults: Thresholds) -> tuple[Thresholds, bool]:
    """Read one complete settings snapshot, or return defaults when absent."""
    settings_path = Path(path)
    try:
        raw = settings_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return defaults, False
    except OSError as exc:
        logger.warning(
            "runtime settings storage failure operation=read errno=%s",
            exc.errno or "unknown",
        )
        raise RuntimeSettingsError("runtime settings are unavailable") from exc
    except UnicodeError as exc:
        logger.warning(
            "runtime settings storage failure operation=read errno=invalid-encoding"
        )
        raise RuntimeSettingsError("runtime settings are unavailable") from exc
    try:
        document = json.loads(raw)
        if not isinstance(document, dict):
            raise ValueError("settings document is not an object")
        return _validate_persisted(document, defaults), True
    except (RuntimeSettingsError, ValueError, TypeError, OverflowError) as exc:
        logger.warning("runtime settings file is invalid")
        raise RuntimeSettingsError("runtime settings are invalid") from exc


def effective_from_settings(config: Any) -> Thresholds:
    """Read one complete threshold snapshot for an inference request."""
    defaults = defaults_from_settings(config)
    effective, _ = read(config.JEV_SETTINGS_PATH, defaults)
    return effective


def write(path: str, values: dict[str, object]) -> Thresholds:
    """Atomically replace the settings file after validating a full update."""
    thresholds = validate_editable(values)
    settings_path = Path(path)
    parent = settings_path.parent
    try:
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{settings_path.name}.", dir=parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                json.dump(
                    {
                        "detection_enter": thresholds.detection_enter,
                        "detection_stay": thresholds.detection_stay,
                        "review_evidence": thresholds.review_evidence,
                        "review_choice": thresholds.review_choice,
                    },
                    output,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, settings_path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    except OSError as exc:
        logger.warning(
            "runtime settings storage failure operation=write errno=%s",
            exc.errno or "unknown",
        )
        raise RuntimeSettingsError("runtime settings could not be saved") from exc
    return thresholds
