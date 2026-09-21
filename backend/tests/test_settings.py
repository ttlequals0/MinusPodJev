"""Tests for persistent local runtime threshold settings."""

import errno
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import pytest
from app.api import openai
from app.config import settings
from app.services import runtime_settings
from app.services.runtime_settings import Thresholds, read, write


def _headers(token: str = "minus-password") -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    path = tmp_path / "runtime-settings.json"
    monkeypatch.setattr(settings, "JEV_SETTINGS_PATH", str(path))
    monkeypatch.setattr(settings, "MINUSPOD_PASSWORD", "minus-password")
    monkeypatch.setattr(settings, "JEV_ENTER", 0.95)
    monkeypatch.setattr(settings, "JEV_STAY", 0.4)
    monkeypatch.setattr(settings, "JEV_REVIEW_EVIDENCE_THRESHOLD", None)
    monkeypatch.setattr(settings, "JEV_REVIEW_CHOICE_THRESHOLD", None)
    return path


async def test_settings_defaults_and_authenticated_persistence(client, settings_file):
    schema = (await client.get("/api/openapi.json")).json()["paths"]["/api/settings"]["put"]["requestBody"]["content"]["application/json"]["schema"]
    assert set(schema["required"]) == {"detection_enter", "detection_stay", "review_evidence", "review_choice"}
    response = await client.get("/api/settings")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "thresholds": {
            "detection_enter": 0.95,
            "detection_stay": 0.4,
            "review_evidence": 0.95,
            "review_choice": 0.95,
        },
        "defaults": {
            "detection_enter": 0.95,
            "detection_stay": 0.4,
            "review_evidence": 0.95,
            "review_choice": 0.95,
        },
        "persisted": False,
        "editable": True,
    }

    update = {"detection_enter": 0.8, "detection_stay": 0.6, "review_evidence": 0.91, "review_choice": 0.87}
    response = await client.put("/api/settings", json=update, headers=_headers())
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["thresholds"] == {
        "detection_enter": 0.8,
        "detection_stay": 0.6,
        "review_evidence": 0.91,
        "review_choice": 0.87,
    }
    assert settings_file.stat().st_mode & 0o777 == 0o600

    response = await client.get("/api/settings")
    assert response.json()["persisted"] is True
    assert response.json()["thresholds"]["detection_enter"] == 0.8


async def test_settings_require_local_password_and_do_not_call_upstream(client, settings_file):
    update = {"detection_enter": 0.8, "detection_stay": 0.6, "review_evidence": 0.91, "review_choice": 0.87}
    assert (await client.put("/api/settings", json=update)).status_code == 403
    assert (await client.put("/api/settings", json=update, headers=_headers("wrong"))).status_code == 403
    assert not settings_file.exists()


@pytest.mark.parametrize(
    "body",
    [
        {"detection_enter": "0.8", "detection_stay": 0.6, "review_evidence": 0.91, "review_choice": 0.87},
        {"detection_enter": True, "detection_stay": 0.6, "review_evidence": 0.91, "review_choice": 0.87},
        {"detection_enter": 0.8, "detection_stay": "0.6", "review_evidence": 0.91, "review_choice": 0.87},
        {"detection_enter": 0.8, "detection_stay": True, "review_evidence": 0.91, "review_choice": 0.87},
        {"detection_enter": 0.8, "detection_stay": -0.01, "review_evidence": 0.91, "review_choice": 0.87},
        {"detection_enter": 0.8, "detection_stay": 1.01, "review_evidence": 0.91, "review_choice": 0.87},
        {"detection_enter": 0.8, "detection_stay": 0.81, "review_evidence": 0.91, "review_choice": 0.87},
        {"detection_enter": 0.8, "detection_stay": 0.6, "review_evidence": 0.91},
        {"detection_enter": 0.8, "detection_stay": 0.6, "review_evidence": 0.91, "review_choice": 0.87, "extra": 1},
        {"detection_enter": 10**400, "detection_stay": 0.6, "review_evidence": 0.91, "review_choice": 0.87},
    ],
)
async def test_settings_reject_invalid_updates(client, settings_file, body):
    response = await client.put("/api/settings", json=body, headers=_headers())
    assert response.status_code == 422
    assert not settings_file.exists()


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity"])
async def test_settings_reject_nonfinite_json(client, settings_file, number):
    content = f'{{"detection_enter":{number},"detection_stay":0.6,"review_evidence":0.91,"review_choice":0.87}}'
    response = await client.put(
        "/api/settings",
        content=content,
        headers={**_headers(), "content-type": "application/json"},
    )
    assert response.status_code == 422
    assert not settings_file.exists()


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity"])
async def test_settings_reject_nonfinite_detection_stay_json(client, settings_file, number):
    content = f'{{"detection_enter":0.8,"detection_stay":{number},"review_evidence":0.91,"review_choice":0.87}}'
    response = await client.put(
        "/api/settings",
        content=content,
        headers={**_headers(), "content-type": "application/json"},
    )
    assert response.status_code == 422
    assert not settings_file.exists()


async def test_settings_corruption_is_explicit_and_put_can_repair(client, settings_file):
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    settings_file.write_bytes(b"not utf-8: \xff")
    response = await client.get("/api/settings")
    assert response.status_code == 503
    assert "not utf-8" not in response.text

    update = {"detection_enter": 0.82, "detection_stay": 0.6, "review_evidence": 0.92, "review_choice": 0.88}
    response = await client.put("/api/settings", json=update, headers=_headers())
    assert response.status_code == 200
    assert json.loads(settings_file.read_text()) == update


async def test_settings_read_storage_error_logs_safe_diagnostic(client, settings_file, caplog):
    settings_file.mkdir()
    with caplog.at_level(logging.WARNING, logger=runtime_settings.__name__):
        response = await client.get("/api/settings")
    assert response.status_code == 503
    assert "runtime settings storage failure operation=read" in caplog.text
    assert f"errno={errno.EISDIR}" in caplog.text
    assert str(settings_file) not in caplog.text


async def test_settings_write_directory_error_is_safe(client, settings_file):
    settings_file.mkdir()
    update = {"detection_enter": 0.82, "detection_stay": 0.6, "review_evidence": 0.92, "review_choice": 0.88}
    response = await client.put("/api/settings", json=update, headers=_headers())
    assert response.status_code == 503
    assert "runtime-settings" not in response.text


async def test_settings_write_storage_error_logs_safe_diagnostic(client, settings_file, caplog, monkeypatch):
    update = {"detection_enter": 0.82, "detection_stay": 0.6, "review_evidence": 0.92, "review_choice": 0.88}
    def fail_replace(*_args):
        raise PermissionError(errno.EACCES, "permission denied", str(settings_file))

    monkeypatch.setattr(runtime_settings.os, "replace", fail_replace)
    with caplog.at_level(logging.WARNING, logger=runtime_settings.__name__):
        response = await client.put("/api/settings", json=update, headers=_headers())
    assert response.status_code == 503
    assert "runtime settings storage failure operation=write" in caplog.text
    assert f"errno={errno.EACCES}" in caplog.text
    assert str(settings_file) not in caplog.text
    assert "permission denied" not in caplog.text
    assert "runtime-settings" not in response.text


async def test_failed_replace_preserves_previous_settings(client, settings_file, monkeypatch):
    original = {"detection_enter": 0.82, "detection_stay": 0.6, "review_evidence": 0.92, "review_choice": 0.88}
    assert (await client.put("/api/settings", json=original, headers=_headers())).status_code == 200

    def fail_replace(*_args):
        raise OSError("replace failed")

    monkeypatch.setattr(runtime_settings.os, "replace", fail_replace)
    replacement = {"detection_enter": 0.83, "detection_stay": 0.61, "review_evidence": 0.93, "review_choice": 0.89}
    response = await client.put("/api/settings", json=replacement, headers=_headers())
    assert response.status_code == 503
    assert (await client.get("/api/settings")).json()["thresholds"]["detection_enter"] == 0.82
    assert list(settings_file.parent.glob(f".{settings_file.name}.*")) == []


async def test_failed_threshold_relationship_update_preserves_previous_settings(client, settings_file):
    original = {"detection_enter": 0.82, "detection_stay": 0.6, "review_evidence": 0.92, "review_choice": 0.88}
    assert (await client.put("/api/settings", json=original, headers=_headers())).status_code == 200

    invalid = {"detection_enter": 0.6, "detection_stay": 0.61, "review_evidence": 0.93, "review_choice": 0.89}
    response = await client.put("/api/settings", json=invalid, headers=_headers())

    assert response.status_code == 422
    assert (await client.get("/api/settings")).json()["thresholds"] == original


async def test_four_field_file_ignores_changed_startup_stay(client, settings_file, monkeypatch):
    update = {"detection_enter": 0.81, "detection_stay": 0.62, "review_evidence": 0.86, "review_choice": 0.88}
    assert (await client.put("/api/settings", json=update, headers=_headers())).status_code == 200
    monkeypatch.setattr(settings, "JEV_STAY", 0.7)

    response = await client.get("/api/settings")

    assert response.status_code == 200
    assert response.json()["thresholds"]["detection_stay"] == 0.62


async def test_inference_takes_one_threshold_snapshot(client, settings_file, monkeypatch):
    update = {"detection_enter": 0.81, "detection_stay": 0.62, "review_evidence": 0.86, "review_choice": 0.88}
    assert (await client.put("/api/settings", json=update, headers=_headers())).status_code == 200
    monkeypatch.setattr(settings, "TYPESAFE_API_KEY", "test-key")
    monkeypatch.setattr(settings, "JEV_ALLOW_UNAUTHENTICATED_FALLBACK", True)
    captured = {}

    def fake_run_chat_completion(**kwargs):
        captured.update(kwargs)
        settings_file.write_text(
            json.dumps({"detection_enter": 0.7, "review_evidence": 0.71, "review_choice": 0.72})
        )
        return {"ok": True}

    monkeypatch.setattr(openai, "run_chat_completion", fake_run_chat_completion)
    response = await client.post(
        "/chat/completions",
        json={"messages": [{"role": "user", "content": "[0s - 1s] ordinary text"}]},
    )
    assert response.status_code == 200
    assert (captured["enter"], captured["stay"]) == (0.81, 0.62)
    assert (captured["review_evidence_enter"], captured["review_choice_enter"]) == (0.86, 0.88)


def test_native_endpoint_uses_persisted_detection_stay(settings_file, monkeypatch):
    import app.api.v1.jev as api

    update = {"detection_enter": 0.81, "detection_stay": 0.62, "review_evidence": 0.86, "review_choice": 0.88}
    write(str(settings_file), update)
    seen = {}

    def fake_ask(_segments, **kwargs):
        seen.update(kwargs)
        return {"probabilities": {}, "spans": [], "usage": {}}

    monkeypatch.setattr(api, "jev_ask", fake_ask)
    result = api.ask(
        api.AskRequest(segments=[{"sid": 1, "text": "hi"}]),
        "Bearer test-key",
    )

    assert result["spans"] == []
    assert (seen["enter"], seen["stay"]) == (0.81, 0.62)


def test_atomic_full_writes_leave_one_valid_snapshot(settings_file):
    defaults = Thresholds(0.95, 0.4, 0.95, 0.95)
    values = [
        {"detection_enter": 0.81 + index / 100, "detection_stay": 0.6, "review_evidence": 0.86, "review_choice": 0.88}
        for index in range(8)
    ]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda item: write(str(settings_file), item), values))
    effective, persisted = read(str(settings_file), defaults)
    assert persisted is True
    assert effective.detection_enter in {item["detection_enter"] for item in values}


def test_read_legacy_three_field_file_uses_startup_stay(settings_file):
    defaults = Thresholds(0.95, 0.4, 0.95, 0.95)
    settings_file.write_text(
        json.dumps({"detection_enter": 0.81, "review_evidence": 0.86, "review_choice": 0.88})
    )

    effective, persisted = read(str(settings_file), defaults)

    assert persisted is True
    assert effective == Thresholds(0.81, 0.4, 0.86, 0.88)
