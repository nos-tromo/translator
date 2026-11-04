"""Tests for translator's data-protection audit events (:mod:`translator.audit`).

Every audited request must write exactly one event with the right action,
operation, outcome, actor, criteria and recipient, and none of the text or the
translation. The ``audit_emitter`` fixture verifies the hash chain after each
test. All names, texts and references are synthetic.
"""

import os
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from loguru import logger
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from tests.conftest import CapturedAuditStream
from translator import main
from translator.audit import EMITTER_STATE_KEY, OUTPUT_ENV, REQUIRE_PURPOSE_ENV, SERVICE, audited, get_required_actions
from translator.main import app
from translator.utils.audit_event import AuditEmitter, sha256_text, verify_chain
from translator.utils.audit_runtime import PURPOSE_HEADER, encode_purpose

_SECRET = "zz-synthetic-edge-secret-0123456789abcdef"
_USER = "zz-alice"
_DEV_IDENTITY = "zz-dev-user"
_REQUEST_ID = "0192c3d4-5e6f-7a8b-9c0d-1e2f3a4b5c6d"
_CASE_REF = "case-0001"
_REASON = "Quality review zz-0001 (Prüfung)"
_SOURCE_TEXT = "Hello Alice Example, see you at zz-synthetic-place-4817."
_TRANSLATION = "Bonjour Alice Example, à bientôt à zz-synthetic-place-4817."
_BODY = {"text": _SOURCE_TEXT, "target_lang": "fr"}


@pytest.fixture
def edge_headers(set_edge_secret: Callable[[str | None], None]) -> dict[str, str]:
    """Enforce the edge secret and return what the gateway sends for a signed-in user.

    Args:
        set_edge_secret: Fixture that sets or unsets the edge secret.

    Returns:
        dict[str, str]: The identity, edge-secret and correlation headers.
    """
    set_edge_secret(_SECRET)
    return {"X-Auth-User": _USER, "X-Edge-Secret": _SECRET, "X-Request-ID": _REQUEST_ID}


@pytest.fixture
def engine() -> Iterator[MagicMock]:
    """Replace the app's engine with a mock that detects English and returns ``_TRANSLATION``.

    Yields:
        MagicMock: The mock engine, for call inspection.
    """
    with patch("translator.main.translator", model="test-model") as mock:
        mock.detect_language.return_value = {"code": "en", "name": "English", "flag": ""}
        mock.get_language_info.return_value = {"name": "English", "flag": ""}
        mock.translate.return_value = _TRANSLATION
        yield mock


def _reply(content: str, *, prompt_tokens: int, completion_tokens: int) -> ChatCompletion:
    """Build a chat-completions reply as the OpenAI client returns it.

    Args:
        content: The message content.
        prompt_tokens: The prompt tokens the endpoint reports.
        completion_tokens: The completion tokens the endpoint reports.

    Returns:
        ChatCompletion: The reply, naming ``zz-served-model`` as the model that served.
    """
    usage = CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    message = ChatCompletionMessage(role="assistant", content=content)
    return ChatCompletion(
        id="chatcmpl-zz-synthetic",
        object="chat.completion",
        created=0,
        model="zz-served-model",
        choices=[Choice(index=0, finish_reason="stop", message=message)],
        usage=usage,
    )


# ── What one translation records ───────────────────────────────────────────────


def test_translation_writes_one_ai_call_event(
    client: TestClient, audit_stream: CapturedAuditStream, engine: MagicMock, edge_headers: dict[str, str]
) -> None:
    """A translation is recorded as one ``ai_call`` to the inference service.

    Args:
        client: TestClient provided by the ``client`` fixture.
        audit_stream: The captured audit stream.
        engine: The mock engine.
        edge_headers: The gateway's headers for a signed-in user.
    """
    r = client.post("/api/v1/translate", headers=edge_headers, json=_BODY)

    assert r.status_code == 200
    assert r.json()["translation"] == _TRANSLATION
    [event] = audit_stream.operation_events()
    assert (event["action"], event["operation"], event["outcome"]) == ("ai_call", "translate", "ok")
    assert event["service"] == SERVICE
    assert event["actor"] == {"user": _USER, "via": "edge"}
    assert event["recipient"] == "inference:translate"
    assert event["request_id"] == _REQUEST_ID
    assert event["criteria"] == {
        "source_lang": None,
        "target_lang": "fr",
        "input_chars": len(_SOURCE_TEXT),
        "detected_lang": "en",
    }
    assert event["ai"] == {"model_alias": "test-model", "output_sha256": sha256_text(_TRANSLATION)}
    assert "target" not in event
    assert "purpose" not in event


def test_events_hold_neither_the_text_nor_the_translation(
    client: TestClient, audit_stream: CapturedAuditStream, engine: MagicMock, edge_headers: dict[str, str]
) -> None:
    """No part of the text or the translation reaches the audit stream.

    Args:
        client: TestClient provided by the ``client`` fixture.
        audit_stream: The captured audit stream.
        engine: The mock engine.
        edge_headers: The gateway's headers for a signed-in user.
    """
    client.post("/api/v1/translate", headers=edge_headers, json=_BODY)
    engine.translate.side_effect = RuntimeError("Translation failed")
    client.post("/api/v1/translate", headers=edge_headers, json=_BODY)

    assert [event["outcome"] for event in audit_stream.operation_events()] == ["ok", "error"]
    raw = audit_stream.getvalue()
    for content in (_SOURCE_TEXT, _TRANSLATION, "Alice Example", "zz-synthetic-place-4817"):
        assert content not in raw


def test_explicit_source_language_is_recorded_as_given(
    client: TestClient, audit_stream: CapturedAuditStream, engine: MagicMock, edge_headers: dict[str, str]
) -> None:
    """With an explicit source language there is no detection call and no detected language.

    Args:
        client: TestClient provided by the ``client`` fixture.
        audit_stream: The captured audit stream.
        engine: The mock engine.
        edge_headers: The gateway's headers for a signed-in user.
    """
    r = client.post("/api/v1/translate", headers=edge_headers, json={**_BODY, "source_lang": "en"})

    assert r.status_code == 200
    engine.detect_language.assert_not_called()
    [event] = audit_stream.operation_events()
    assert event["criteria"] == {"source_lang": "en", "target_lang": "fr", "input_chars": len(_SOURCE_TEXT)}


def test_failed_detection_is_recorded_as_no_detected_language(
    client: TestClient, audit_stream: CapturedAuditStream, engine: MagicMock, edge_headers: dict[str, str]
) -> None:
    """When the model cannot name the source language, the event says so.

    Args:
        client: TestClient provided by the ``client`` fixture.
        audit_stream: The captured audit stream.
        engine: The mock engine.
        edge_headers: The gateway's headers for a signed-in user.
    """
    engine.detect_language.return_value = {"code": "", "name": "", "flag": ""}

    client.post("/api/v1/translate", headers=edge_headers, json=_BODY)

    [event] = audit_stream.operation_events()
    assert event["outcome"] == "ok"
    assert event["criteria"]["detected_lang"] is None


def test_failed_translation_writes_one_error_event(
    client: TestClient, audit_stream: CapturedAuditStream, engine: MagicMock, edge_headers: dict[str, str]
) -> None:
    """A failed translation is recorded as an ``error``, without an output hash.

    Args:
        client: TestClient provided by the ``client`` fixture.
        audit_stream: The captured audit stream.
        engine: The mock engine.
        edge_headers: The gateway's headers for a signed-in user.
    """
    engine.translate.side_effect = RuntimeError("Translation failed")

    r = client.post("/api/v1/translate", headers=edge_headers, json=_BODY)

    assert r.status_code == 500
    assert r.json() == {"detail": "Translation failed."}
    [event] = audit_stream.operation_events()
    assert (event["action"], event["operation"], event["outcome"]) == ("ai_call", "translate", "error")
    assert event["recipient"] == "inference:translate"
    assert event["criteria"]["target_lang"] == "fr"
    assert event["ai"] == {"model_alias": "test-model"}


def test_event_carries_reported_usage_and_upstream_gets_only_the_request_id(
    client: TestClient, audit_stream: CapturedAuditStream, edge_headers: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the real engine, the event sums both calls' usage; upstream sees the correlation id only.

    Args:
        client: TestClient provided by the ``client`` fixture.
        audit_stream: The captured audit stream.
        edge_headers: The gateway's headers for a signed-in user.
        monkeypatch: Pytest fixture replacing the engine's OpenAI client.
    """
    upstream = MagicMock()
    upstream.chat.completions.create.side_effect = [
        _reply("en", prompt_tokens=40, completion_tokens=1),
        _reply(_TRANSLATION, prompt_tokens=60, completion_tokens=20),
    ]
    monkeypatch.setattr(main.translator, "client", upstream)
    headers = {**edge_headers, PURPOSE_HEADER: encode_purpose(_CASE_REF, _REASON)}

    r = client.post("/api/v1/translate", headers=headers, json=_BODY)

    assert r.status_code == 200
    [event] = audit_stream.operation_events()
    assert event["ai"] == {
        "model_alias": main.translator.model,
        "model_id": "zz-served-model",
        "output_sha256": sha256_text(_TRANSLATION),
        "input_tokens": 100,
        "output_tokens": 21,
    }
    calls = upstream.chat.completions.create.call_args_list
    assert [call.kwargs["extra_headers"] for call in calls] == [{"X-Request-ID": _REQUEST_ID}] * 2
    sent = repr(calls)
    for value in (_USER, _SECRET, _CASE_REF, "Quality review"):
        assert value not in sent


# ── Fail closed ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("breakage", ["stream-fails", "emitter-closed"])
def test_unrecorded_translation_is_not_returned(
    client: TestClient,
    audit_stream: CapturedAuditStream,
    audit_emitter: AuditEmitter,
    engine: MagicMock,
    edge_headers: dict[str, str],
    breakage: str,
) -> None:
    """If the event cannot be written, the request fails and the translation stays back.

    Args:
        client: TestClient provided by the ``client`` fixture.
        audit_stream: The captured audit stream.
        audit_emitter: The installed emitter.
        engine: The mock engine.
        edge_headers: The gateway's headers for a signed-in user.
        breakage: How the audit output is broken.
    """
    if breakage == "stream-fails":
        audit_stream.failing = True
    else:
        audit_emitter.close()

    r = client.post("/api/v1/translate", headers=edge_headers, json=_BODY)

    assert r.status_code == 500
    assert r.json() == {"detail": "Translation failed."}
    assert _TRANSLATION not in r.text
    assert audit_stream.operation_events() == []


def test_translate_without_an_emitter_is_refused(engine: MagicMock, edge_headers: dict[str, str]) -> None:
    """Without the process's emitter, audited work never starts.

    Args:
        engine: The mock engine.
        edge_headers: The gateway's headers for a signed-in user.
    """
    r = TestClient(app).post("/api/v1/translate", headers=edge_headers, json=_BODY)

    assert r.status_code == 500
    engine.detect_language.assert_not_called()
    engine.translate.assert_not_called()


# ── Who acts ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("secret", "headers", "default_identity", "actor"),
    [
        (_SECRET, {"X-Auth-User": _USER, "X-Edge-Secret": _SECRET}, None, {"user": _USER, "via": "edge"}),
        (None, {"X-Auth-User": _USER}, None, {"user": _USER, "via": "edge"}),
        (None, {"X-Auth-User": _USER}, _DEV_IDENTITY, {"user": _USER, "via": "edge"}),
        (None, {}, _DEV_IDENTITY, {"user": _DEV_IDENTITY, "via": "system"}),
    ],
    ids=["edge-verified", "edge-secret-unset", "header-wins-over-default", "default-identity"],
)
def test_actor_follows_the_principal(
    client: TestClient,
    audit_stream: CapturedAuditStream,
    engine: MagicMock,
    set_edge_secret: Callable[[str | None], None],
    set_default_identity: Callable[[str | None], None],
    secret: str | None,
    headers: dict[str, str],
    default_identity: str | None,
    actor: dict[str, str],
) -> None:
    """A gateway-asserted account acts via ``edge``; the dev default identity via ``system``.

    Args:
        client: TestClient provided by the ``client`` fixture.
        audit_stream: The captured audit stream.
        engine: The mock engine.
        set_edge_secret: Fixture that sets or unsets the edge secret.
        set_default_identity: Fixture that sets or unsets the default identity.
        secret: The configured edge secret, or ``None``.
        headers: The request's identity headers.
        default_identity: The configured default identity, or ``None``.
        actor: The expected actor.
    """
    set_edge_secret(secret)
    set_default_identity(default_identity)

    r = client.post("/api/v1/translate", headers=headers, json=_BODY)

    assert r.status_code == 200
    [event] = audit_stream.operation_events()
    assert event["actor"] == actor


@pytest.mark.parametrize(
    ("secret", "headers", "default_identity"),
    [
        (None, {}, None),
        (_SECRET, {}, None),
        (_SECRET, {"X-Auth-User": _USER}, None),
        (_SECRET, {"X-Auth-User": _USER, "X-Edge-Secret": "zz-wrong"}, None),
        (_SECRET, {"X-Auth-User": _USER, "X-Edge-Secret": "zz-wrong"}, _DEV_IDENTITY),
        (_SECRET, {}, _DEV_IDENTITY),
    ],
    ids=[
        "no-identity",
        "edge-secret-set-no-identity",
        "edge-secret-missing",
        "edge-secret-wrong",
        "unverified-with-default-identity",
        "default-identity-ignored",
    ],
)
def test_unauthenticated_request_is_refused_and_not_recorded(
    client: TestClient,
    audit_stream: CapturedAuditStream,
    engine: MagicMock,
    set_edge_secret: Callable[[str | None], None],
    set_default_identity: Callable[[str | None], None],
    secret: str | None,
    headers: dict[str, str],
    default_identity: str | None,
) -> None:
    """A request without a trusted identity gets 401 and leaves no event.

    While the edge secret is set, a configured default identity does not
    change that.

    Args:
        client: TestClient provided by the ``client`` fixture.
        audit_stream: The captured audit stream.
        engine: The mock engine.
        set_edge_secret: Fixture that sets or unsets the edge secret.
        set_default_identity: Fixture that sets or unsets the default identity.
        secret: The configured edge secret, or ``None``.
        headers: The request's identity headers.
        default_identity: The configured default identity, or ``None``.
    """
    set_edge_secret(secret)
    set_default_identity(default_identity)

    r = client.post("/api/v1/translate", headers=headers, json=_BODY)

    assert r.status_code == 401
    assert audit_stream.operation_events() == []
    engine.translate.assert_not_called()


# ── Why: the stated purpose ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("case_ref", "reason", "recorded"),
    [
        (_CASE_REF, _REASON, {"case_ref": _CASE_REF, "reason": _REASON}),
        (_CASE_REF, None, {"case_ref": _CASE_REF}),
        (None, _REASON, {"reason": _REASON}),
    ],
    ids=["both", "case-ref-only", "reason-only"],
)
def test_stated_purpose_is_recorded(
    client: TestClient,
    audit_stream: CapturedAuditStream,
    engine: MagicMock,
    edge_headers: dict[str, str],
    case_ref: str | None,
    reason: str | None,
    recorded: dict[str, str],
) -> None:
    """The ``X-Audit-Purpose`` header ends up, decoded, in the event.

    Args:
        client: TestClient provided by the ``client`` fixture.
        audit_stream: The captured audit stream.
        engine: The mock engine.
        edge_headers: The gateway's headers for a signed-in user.
        case_ref: The case reference sent.
        reason: The reason sent.
        recorded: The expected ``purpose`` of the event.
    """
    headers = {**edge_headers, PURPOSE_HEADER: encode_purpose(case_ref, reason)}

    r = client.post("/api/v1/translate", headers=headers, json=_BODY)

    assert r.status_code == 200
    [event] = audit_stream.operation_events()
    assert event["purpose"] == recorded


def test_stated_purpose_stays_out_of_ops_logs(
    client: TestClient, audit_stream: CapturedAuditStream, engine: MagicMock, edge_headers: dict[str, str]
) -> None:
    """The purpose is personal data in its own right: no log line carries it, not even on failure.

    Args:
        client: TestClient provided by the ``client`` fixture.
        audit_stream: The captured audit stream.
        engine: The mock engine.
        edge_headers: The gateway's headers for a signed-in user.
    """
    headers = {**edge_headers, PURPOSE_HEADER: encode_purpose(_CASE_REF, _REASON)}
    messages: list[str] = []
    handler_id = logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        client.post("/api/v1/translate", headers=headers, json=_BODY)
        engine.translate.side_effect = RuntimeError("Translation failed")
        client.post("/api/v1/translate", headers=headers, json=_BODY)
        client.post("/api/v1/translate", headers={**headers, PURPOSE_HEADER: f"case_ref={_CASE_REF}%0Ax"}, json=_BODY)
    finally:
        logger.remove(handler_id)

    assert len(audit_stream.operation_events()) == 2
    logged = "".join(messages)
    for value in (_CASE_REF, "Quality review"):
        assert value not in logged


@pytest.mark.parametrize(
    "value",
    [
        "case_ref=a&case_ref=b",
        "case_ref=" + "x" * 257,
        "reason=%FF",
        "case_ref=line%0Abreak",
        "case_ref",
        "x" * 4097,
    ],
    ids=["repeated-field", "field-too-long", "not-utf8", "control-character", "not-form-encoded", "header-too-long"],
)
def test_malformed_purpose_is_refused_with_400_and_not_recorded(
    client: TestClient,
    audit_stream: CapturedAuditStream,
    engine: MagicMock,
    edge_headers: dict[str, str],
    value: str,
) -> None:
    """A purpose that cannot be read unambiguously gets 400, before any work and any event.

    Args:
        client: TestClient provided by the ``client`` fixture.
        audit_stream: The captured audit stream.
        engine: The mock engine.
        edge_headers: The gateway's headers for a signed-in user.
        value: The malformed header value.
    """
    r = client.post("/api/v1/translate", headers={**edge_headers, PURPOSE_HEADER: value}, json=_BODY)

    assert r.status_code == 400
    assert r.json() == {"detail": "Invalid X-Audit-Purpose header."}
    assert audit_stream.operation_events() == []
    engine.translate.assert_not_called()


def test_repeated_purpose_header_is_refused_with_400(
    client: TestClient, audit_stream: CapturedAuditStream, engine: MagicMock, edge_headers: dict[str, str]
) -> None:
    """Two ``X-Audit-Purpose`` headers are ambiguous, so the request is refused.

    Args:
        client: TestClient provided by the ``client`` fixture.
        audit_stream: The captured audit stream.
        engine: The mock engine.
        edge_headers: The gateway's headers for a signed-in user.
    """
    headers = [*edge_headers.items(), (PURPOSE_HEADER, "case_ref=case-0001"), (PURPOSE_HEADER, "case_ref=case-0002")]

    r = client.post("/api/v1/translate", headers=headers, json=_BODY)

    assert r.status_code == 400
    assert audit_stream.operation_events() == []
    engine.translate.assert_not_called()


def test_missing_required_purpose_is_refused_with_428_and_recorded(
    client: TestClient,
    audit_stream: CapturedAuditStream,
    engine: MagicMock,
    edge_headers: dict[str, str],
    set_required_actions: Callable[[str | None], None],
) -> None:
    """With ``ai_call`` required, a translation without a purpose is denied and recorded once.

    Args:
        client: TestClient provided by the ``client`` fixture.
        audit_stream: The captured audit stream.
        engine: The mock engine.
        edge_headers: The gateway's headers for a signed-in user.
        set_required_actions: Fixture that sets or unsets ``AUDIT_REQUIRE_PURPOSE``.
    """
    set_required_actions("ai_call")

    r = client.post("/api/v1/translate", headers=edge_headers, json=_BODY)

    assert r.status_code == 428
    assert r.json() == {"detail": "This action requires a stated purpose."}
    engine.detect_language.assert_not_called()
    engine.translate.assert_not_called()
    [event] = audit_stream.operation_events()
    assert (event["action"], event["operation"], event["outcome"]) == ("ai_call", "translate", "denied")
    assert event["actor"] == {"user": _USER, "via": "edge"}
    assert event["request_id"] == _REQUEST_ID
    assert event["criteria"] == {"source_lang": None, "target_lang": "fr", "input_chars": len(_SOURCE_TEXT)}
    for absent in ("purpose", "target", "recipient", "ai"):
        assert absent not in event


def test_required_purpose_when_stated_lets_the_translation_through(
    client: TestClient,
    audit_stream: CapturedAuditStream,
    engine: MagicMock,
    edge_headers: dict[str, str],
    set_required_actions: Callable[[str | None], None],
) -> None:
    """With ``ai_call`` required, a stated purpose is enough.

    Args:
        client: TestClient provided by the ``client`` fixture.
        audit_stream: The captured audit stream.
        engine: The mock engine.
        edge_headers: The gateway's headers for a signed-in user.
        set_required_actions: Fixture that sets or unsets ``AUDIT_REQUIRE_PURPOSE``.
    """
    set_required_actions("ai_call")
    headers = {**edge_headers, PURPOSE_HEADER: encode_purpose(_CASE_REF, None)}

    r = client.post("/api/v1/translate", headers=headers, json=_BODY)

    assert r.status_code == 200
    [event] = audit_stream.operation_events()
    assert (event["outcome"], event["purpose"]) == ("ok", {"case_ref": _CASE_REF})


@pytest.mark.parametrize("required", [None, "", "query, disclose"], ids=["unset", "empty", "other-actions"])
def test_no_428_unless_ai_call_is_required(
    client: TestClient,
    audit_stream: CapturedAuditStream,
    engine: MagicMock,
    edge_headers: dict[str, str],
    set_required_actions: Callable[[str | None], None],
    required: str | None,
) -> None:
    """Without ``ai_call`` in ``AUDIT_REQUIRE_PURPOSE``, no purpose is needed.

    Args:
        client: TestClient provided by the ``client`` fixture.
        audit_stream: The captured audit stream.
        engine: The mock engine.
        edge_headers: The gateway's headers for a signed-in user.
        set_required_actions: Fixture that sets or unsets ``AUDIT_REQUIRE_PURPOSE``.
        required: The configured value.
    """
    set_required_actions(required)

    r = client.post("/api/v1/translate", headers=edge_headers, json=_BODY)

    assert r.status_code == 200
    [event] = audit_stream.operation_events()
    assert event["outcome"] == "ok"


def test_unrecorded_refusal_fails_closed(
    client: TestClient,
    audit_stream: CapturedAuditStream,
    engine: MagicMock,
    edge_headers: dict[str, str],
    set_required_actions: Callable[[str | None], None],
) -> None:
    """If the denied event cannot be written, the answer is a generic 500.

    Args:
        client: TestClient provided by the ``client`` fixture.
        audit_stream: The captured audit stream.
        engine: The mock engine.
        edge_headers: The gateway's headers for a signed-in user.
        set_required_actions: Fixture that sets or unsets ``AUDIT_REQUIRE_PURPOSE``.
    """
    set_required_actions("ai_call")
    audit_stream.failing = True

    r = client.post("/api/v1/translate", headers=edge_headers, json=_BODY)

    assert r.status_code == 500
    assert r.json() == {"detail": "Request failed."}
    engine.translate.assert_not_called()


def test_required_actions_accept_every_action_but_lifecycle(set_required_actions: Callable[[str | None], None]) -> None:
    """``AUDIT_REQUIRE_PURPOSE`` is case-insensitive, ignores blanks and refuses unknown names.

    Args:
        set_required_actions: Fixture that sets or unsets ``AUDIT_REQUIRE_PURPOSE``.
    """
    set_required_actions(" AI_CALL , query,, ")
    assert get_required_actions() == frozenset({"ai_call", "query"})
    set_required_actions("lifecycle")
    with pytest.raises(ValueError, match="lifecycle"):
        get_required_actions()


def test_unknown_required_action_stops_app_startup(tmp_path: Path) -> None:
    """A misspelt action fails the import, as uvicorn's import of the app would.

    Args:
        tmp_path: Empty working directory for the child process.
    """
    env = {
        **os.environ,
        "OPENAI_API_BASE": "http://test-host:11434/v1",
        "TEXT_MODEL": "test-model",
        "TRANSLATOR_EDGE_SECRET": "",
        "TRANSLATOR_DEFAULT_IDENTITY": "",
        REQUIRE_PURPOSE_ENV: "ai_call, ai-call",
    }
    result = subprocess.run(
        [sys.executable, "-c", "import translator.main"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode != 0
    assert "unknown audit action(s) in AUDIT_REQUIRE_PURPOSE: ai-call" in result.stderr


# ── Correlation ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "request_id", ["zz bad id", "x" * 129, "-leading-dash", "zzé"], ids=["space", "long", "dash", "non-ascii"]
)
def test_implausible_request_id_is_neither_recorded_nor_forwarded(
    client: TestClient,
    audit_stream: CapturedAuditStream,
    engine: MagicMock,
    edge_headers: dict[str, str],
    request_id: str,
) -> None:
    """Only a sane correlation id is kept; anything else is dropped, not cleaned up.

    Args:
        client: TestClient provided by the ``client`` fixture.
        audit_stream: The captured audit stream.
        engine: The mock engine.
        edge_headers: The gateway's headers for a signed-in user.
        request_id: The implausible ``X-Request-ID`` value.
    """
    headers = {**edge_headers, "X-Request-ID": request_id}

    r = client.post("/api/v1/translate", headers={k.encode(): v.encode() for k, v in headers.items()}, json=_BODY)

    assert r.status_code == 200
    [event] = audit_stream.operation_events()
    assert "request_id" not in event
    assert engine.translate.call_args.kwargs["trace"].request_id is None


# ── What is not audited ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "path", ["/api/v1/languages", "/api/v1/version", "/api/v1/config", "/api/v1/whoami", "/api/v1/health", "/metrics"]
)
def test_unaudited_endpoints_write_no_event(
    client: TestClient, audit_stream: CapturedAuditStream, edge_headers: dict[str, str], path: str
) -> None:
    """Endpoints that return no personal data and transfer nothing write no event.

    Args:
        client: TestClient provided by the ``client`` fixture.
        audit_stream: The captured audit stream.
        edge_headers: The gateway's headers for a signed-in user.
        path: The endpoint.
    """
    assert client.get(path, headers=edge_headers).status_code == 200
    assert audit_stream.operation_events() == []


@pytest.mark.parametrize("action", ["lifecycle", "translate"])
def test_audited_refuses_a_non_endpoint_action(action: str) -> None:
    """An endpoint can only be audited under one of the library's operation actions.

    Args:
        action: An action name no endpoint may record.
    """
    with pytest.raises(ValueError, match=action):
        audited(action, "translate")


# ── The process's chain ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("secret", "default_identity", "required", "mode"),
    [
        (None, None, None, {"edge_secret_enforced": False, "default_identity_enabled": False, "require_purpose": []}),
        (
            _SECRET,
            None,
            "query, ai_call",
            {"edge_secret_enforced": True, "default_identity_enabled": False, "require_purpose": ["ai_call", "query"]},
        ),
        (
            None,
            _DEV_IDENTITY,
            None,
            {"edge_secret_enforced": False, "default_identity_enabled": True, "require_purpose": []},
        ),
        (
            _SECRET,
            _DEV_IDENTITY,
            None,
            {"edge_secret_enforced": True, "default_identity_enabled": False, "require_purpose": []},
        ),
    ],
    ids=["defaults", "edge-secret-and-purpose", "dev-default-identity", "default-identity-ignored"],
)
def test_lifespan_runs_one_chain_and_records_the_operating_mode(
    monkeypatch: pytest.MonkeyPatch,
    engine: MagicMock,
    set_edge_secret: Callable[[str | None], None],
    set_default_identity: Callable[[str | None], None],
    set_required_actions: Callable[[str | None], None],
    secret: str | None,
    default_identity: str | None,
    required: str | None,
    mode: dict[str, object],
) -> None:
    """The lifespan opens the chain, records the operating mode, serves on it and closes it.

    Args:
        monkeypatch: Pytest fixture redirecting the lifespan's emitter to memory.
        engine: The mock engine.
        set_edge_secret: Fixture that sets or unsets the edge secret.
        set_default_identity: Fixture that sets or unsets the default identity.
        set_required_actions: Fixture that sets or unsets ``AUDIT_REQUIRE_PURPOSE``.
        secret: The configured edge secret, or ``None``.
        default_identity: The configured default identity, or ``None``.
        required: The configured ``AUDIT_REQUIRE_PURPOSE``, or ``None``.
        mode: The expected ``config.audit`` criteria.
    """
    set_edge_secret(secret)
    set_default_identity(default_identity)
    set_required_actions(required)
    stream = CapturedAuditStream()

    def _in_memory(service_version: str) -> AuditEmitter:
        return AuditEmitter(SERVICE, service_version, stream=stream, heartbeat_interval_s=0)

    monkeypatch.setattr("translator.main.open_emitter", _in_memory)
    monkeypatch.setattr(app.state, EMITTER_STATE_KEY, None, raising=False)
    headers = {"X-Auth-User": _USER, "X-Edge-Secret": _SECRET, PURPOSE_HEADER: encode_purpose(_CASE_REF, None)}

    with TestClient(app) as lifespan_client:
        assert lifespan_client.post("/api/v1/translate", headers=headers, json=_BODY).status_code == 200

    events = stream.events()
    assert [(event["action"], event["operation"]) for event in events] == [
        ("lifecycle", "start"),
        ("admin", "config.audit"),
        ("ai_call", "translate"),
        ("lifecycle", "shutdown"),
    ]
    assert {event["service_version"] for event in events} == {app.version}
    config = events[1]
    assert config["actor"] == {"user": None, "via": "system"}
    assert config["criteria"] == mode
    for secret_or_account in (_SECRET, _DEV_IDENTITY):
        assert secret_or_account not in stream.getvalue()
    assert verify_chain(events).ok


def test_unknown_audit_output_stops_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ``AUDIT_OUTPUT`` that names no mode fails startup instead of writing nowhere.

    Args:
        monkeypatch: Pytest fixture setting the variable and restoring the app state.
    """
    monkeypatch.setenv(OUTPUT_ENV, "zz-nowhere")
    monkeypatch.setattr(app.state, EMITTER_STATE_KEY, None, raising=False)

    with pytest.raises(ValueError, match="unknown AUDIT_OUTPUT"), TestClient(app):
        pass
