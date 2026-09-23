import json
from typing import Any

import pytest
from app.api import openai
from app.config import settings
from app.services.openai_adapter import ReviewInconclusiveError
from fastapi.responses import JSONResponse


def test_inconclusive_api_error_exposes_only_controlled_diagnostics(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(openai, "is_review_request", lambda *_args: True)
    monkeypatch.setattr(
        openai,
        "effective_from_settings",
        lambda _settings: type(
            "Thresholds",
            (),
            {
                "detection_enter": 0.5,
                "detection_stay": 0.5,
                "review_evidence": 0.7,
                "review_choice": 0.6,
            },
        )(),
    )

    def fail(**_kwargs: Any) -> None:
        raise ReviewInconclusiveError(
            "secret=caller-token transcript=private words",
            reason="proposed_range_not_confirmed",
            stage="focused_validation",
            score=0.42,
            threshold=0.65,
            cache_hit=False,
        )

    monkeypatch.setattr(openai, "run_chat_completion", fail)
    response = openai.chat_completions(
        openai.ChatCompletionRequest(messages=[]), "Bearer caller-token"
    )
    assert isinstance(response, JSONResponse)
    body = bytes(response.body)
    payload = json.loads(body)
    error = payload["error"]

    assert response.status_code == 422
    assert response.headers["x-should-retry"] == "false"
    assert response.headers["x-request-id"]
    assert error["code"] == "jev_review_inconclusive"
    assert error["reason"] == "proposed_range_not_confirmed"
    assert error["stage"] == "focused_validation"
    assert error["score"] == 0.42
    assert error["threshold"] == 0.65
    assert error["cache_hit"] is False
    assert "proposed_range_not_confirmed" in error["message"]
    assert "focused_validation" in error["message"]
    assert "caller-token" not in body.decode()
    assert "private words" not in body.decode()
    assert "caller-token" not in caplog.text
    assert "private words" not in caplog.text


def test_untrusted_exception_attributes_are_filtered_from_response_and_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(openai, "is_review_request", lambda *_args: True)
    monkeypatch.setattr(
        openai,
        "effective_from_settings",
        lambda _settings: type(
            "Thresholds",
            (),
            {
                "detection_enter": 0.5,
                "detection_stay": 0.5,
                "review_evidence": 0.7,
                "review_choice": 0.6,
            },
        )(),
    )

    def fail(**_kwargs: Any) -> None:
        error = ReviewInconclusiveError(
            "private transcript and credential", reason="transcript_gap", stage="context"
        )
        error.reason = "credential-private text"
        error.stage = "transcript-private text"
        error.score = float("nan")
        error.threshold = True
        object.__setattr__(error, "cache_hit", "private text")
        raise error

    monkeypatch.setattr(openai, "run_chat_completion", fail)
    response = openai.chat_completions(
        openai.ChatCompletionRequest(messages=[]), "Bearer test-key"
    )
    assert isinstance(response, JSONResponse)
    body = bytes(response.body)
    payload = json.loads(body)
    error = payload["error"]

    assert error["reason"] == "malformed_context"
    assert error["stage"] == "context"
    assert "score" not in error
    assert "threshold" not in error
    assert "cache_hit" not in error
    assert "credential-private" not in body.decode()
    assert "private text" not in caplog.text
    assert openai._metric_inconclusive_reason(error) == "malformed_context"


def test_new_inconclusive_reasons_map_to_existing_metric_enums() -> None:
    for reason in (
        "invalid_pair",
        "proposed_range_not_confirmed",
        "original_range_not_confirmed",
    ):
        assert openai._metric_inconclusive_reason(
            {"reason": reason, "stage": "focused_validation"}
        ) == "choice_inconclusive"


def test_boundary_coverage_diagnostics_are_whitelisted_and_finite(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(openai, "is_review_request", lambda *_args: True)
    monkeypatch.setattr(
        openai,
        "effective_from_settings",
        lambda _settings: type(
            "Thresholds",
            (),
            {"detection_enter": 0.5, "detection_stay": 0.5, "review_evidence": 0.7, "review_choice": 0.6},
        )(),
    )

    def fail(**_kwargs: Any) -> None:
        error = ReviewInconclusiveError(
            "secret=private words", reason="missing_boundary_coverage", stage="boundary_coverage"
        )
        error.range_start = 12.5
        error.range_end = 21
        error.start_supported = False
        error.end_supported = True
        error.score = float("inf")
        error.threshold = True
        error.cache_hit = "secret"
        error.extra = "private words"
        raise error

    monkeypatch.setattr(openai, "run_chat_completion", fail)
    response = openai.chat_completions(openai.ChatCompletionRequest(messages=[]), "Bearer test-key")
    assert isinstance(response, JSONResponse)
    body = bytes(response.body).decode()
    error = json.loads(body)["error"]
    assert response.status_code == 422
    assert response.headers["x-should-retry"] == "false"
    assert response.headers["x-request-id"]
    assert error["reason"] == "missing_boundary_coverage"
    assert error["stage"] == "boundary_coverage"
    assert error["range_start"] == 12.5
    assert error["range_end"] == 21
    assert error["start_supported"] is False
    assert error["end_supported"] is True
    assert "score" not in error
    assert "threshold" not in error
    assert "cache_hit" not in error
    assert "extra" not in error
    assert "missing_boundary_coverage" in error["message"]
    assert "boundary_coverage" in error["message"]
    assert "private words" not in body
    assert "private words" not in caplog.text
    assert openai._metric_inconclusive_reason(error) == "missing_boundary_coverage"


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), float("-inf"), "12.5"])
def test_boundary_ranges_reject_bool_nonfinite_and_non_numeric_values(value: Any) -> None:
    error = ReviewInconclusiveError(
        "unused", reason="missing_boundary_coverage", stage="boundary_coverage"
    )
    error.range_start = value
    error.range_end = value
    diagnostics = openai._inconclusive_diagnostics(error)
    assert "range_start" not in diagnostics
    assert "range_end" not in diagnostics


@pytest.mark.parametrize("value", [1, "false", None])
def test_boundary_support_flags_require_booleans(value: Any) -> None:
    error = ReviewInconclusiveError(
        "unused", reason="missing_boundary_coverage", stage="boundary_coverage"
    )
    error.start_supported = value
    error.end_supported = value
    diagnostics = openai._inconclusive_diagnostics(error)
    assert "start_supported" not in diagnostics
    assert "end_supported" not in diagnostics
