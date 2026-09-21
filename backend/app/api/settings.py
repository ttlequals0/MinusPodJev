"""Local runtime threshold settings management."""

from __future__ import annotations

import logging
import math
import secrets
from typing import Any

from fastapi import APIRouter, Body, Header, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from app.config import settings
from app.services.runtime_settings import (
    RuntimeSettingsError,
    RuntimeSettingsValidationError,
    Thresholds,
    defaults_from_settings,
    read,
    write,
)

router = APIRouter()
logger = logging.getLogger(__name__)
_NO_STORE = {"Cache-Control": "no-store"}


class ThresholdUpdate(BaseModel):
    """The complete set of editable runtime thresholds."""

    detection_enter: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    detection_stay: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    review_evidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    review_choice: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def reject_non_numeric_values(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise ValueError("thresholds must be an object")
        expected = {"detection_enter", "detection_stay", "review_evidence", "review_choice"}
        if set(value) != expected:
            raise ValueError("thresholds must contain exactly four fields")
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            for item in value.values()
        ):
            raise ValueError("thresholds must contain only numeric values")
        try:
            finite = all(math.isfinite(float(item)) for item in value.values())
        except (OverflowError, ValueError):
            finite = False
        if not finite:
            raise ValueError("thresholds must contain finite values")
        return value


def _authorized(authorization: str | None) -> bool:
    configured = settings.MINUSPOD_PASSWORD
    if not configured or not authorization:
        return False
    scheme, separator, token = authorization.partition(" ")
    if not separator or scheme.lower() != "bearer":
        return False
    return secrets.compare_digest(token.encode("utf-8"), configured.encode("utf-8"))


def _effective() -> tuple[Thresholds, Thresholds, bool]:
    defaults = defaults_from_settings(settings)
    try:
        effective, persisted = read(settings.JEV_SETTINGS_PATH, defaults)
    except RuntimeSettingsError as exc:
        raise HTTPException(
            status_code=503, detail="Runtime settings unavailable", headers=_NO_STORE
        ) from exc
    return effective, defaults, persisted


def _body(effective: Thresholds, defaults: Thresholds, persisted: bool) -> dict[str, Any]:
    return {
        "thresholds": {
            "detection_enter": effective.detection_enter,
            "detection_stay": effective.detection_stay,
            "review_evidence": effective.review_evidence,
            "review_choice": effective.review_choice,
        },
        "defaults": {
            "detection_enter": defaults.detection_enter,
            "detection_stay": defaults.detection_stay,
            "review_evidence": defaults.review_evidence,
            "review_choice": defaults.review_choice,
        },
        "persisted": persisted,
        "editable": bool(settings.MINUSPOD_PASSWORD),
    }


@router.get("/settings")
def get_settings(response: Response) -> dict[str, Any]:
    """Return effective and startup-default threshold settings."""
    response.headers["Cache-Control"] = "no-store"
    effective, defaults, persisted = _effective()
    return _body(effective, defaults, persisted)


@router.put("/settings")
def update_settings(
    response: Response,
    update: object = Body(..., json_schema_extra=ThresholdUpdate.model_json_schema()),
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    """Persist a complete threshold update using the configured local secret."""
    response.headers["Cache-Control"] = "no-store"
    if not _authorized(authorization):
        raise HTTPException(
            status_code=403,
            detail="Runtime settings writes are disabled",
            headers=_NO_STORE,
        )
    try:
        parsed = ThresholdUpdate.model_validate(update)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="Invalid runtime settings", headers=_NO_STORE) from exc
    defaults = defaults_from_settings(settings)
    try:
        effective = write(settings.JEV_SETTINGS_PATH, parsed.model_dump())
    except RuntimeSettingsValidationError as exc:
        raise HTTPException(
            status_code=422, detail="Invalid runtime settings", headers=_NO_STORE
        ) from exc
    except RuntimeSettingsError as exc:
        raise HTTPException(
            status_code=503, detail="Runtime settings unavailable", headers=_NO_STORE
        ) from exc
    logger.info(
        "runtime settings saved detection_enter=%s detection_stay=%s review_evidence=%s review_choice=%s",
        effective.detection_enter,
        effective.detection_stay,
        effective.review_evidence,
        effective.review_choice,
    )
    return _body(effective, defaults, True)
