"""Unit tests for :class:`translator.engine.Translator`.

Each public method and the constructor are covered in isolation. The OpenAI
client is replaced with a :class:`unittest.mock.MagicMock` wherever a network
call would otherwise be made.
"""

import re
from unittest.mock import MagicMock

import pytest
from openai import BadRequestError
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from translator.engine import DETECTION_PREFIX_LIMIT, InferenceTrace, Translator, cache_scope_for


@pytest.fixture
def translator() -> Translator:
    """Return a :class:`Translator` instance using the test ``OPENAI_API_BASE``.

    Returns:
        Translator: Freshly constructed engine ready for patching.
    """
    return Translator()


# ── _create_client ─────────────────────────────────────────────────────────────


def test_create_client_raises_without_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raises ``ValueError`` when ``OPENAI_API_BASE`` is unset.

    Args:
        monkeypatch: Pytest fixture for temporarily removing environment variables.
    """
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_BASE"):
        Translator()


def test_constructor_raises_without_text_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raises ``ValueError`` when ``TEXT_MODEL`` is unset.

    Args:
        monkeypatch: Pytest fixture for temporarily removing environment variables.
    """
    monkeypatch.delenv("TEXT_MODEL", raising=False)
    with pytest.raises(ValueError, match="TEXT_MODEL"):
        Translator()


def test_constructor_reads_text_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Uses the ``TEXT_MODEL`` value verbatim as the model identifier.

    Args:
        monkeypatch: Pytest fixture for setting environment variables.
    """
    monkeypatch.setenv("TEXT_MODEL", "some/model-id")
    assert Translator().model == "some/model-id"


def test_create_client_succeeds_with_base_url(translator: Translator) -> None:
    """``client`` is initialised when ``OPENAI_API_BASE`` is present.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
    """
    assert translator.client is not None


def test_create_client_passes_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Passes ``OPENAI_TIMEOUT`` to the OpenAI client constructor as a float.

    Args:
        monkeypatch: Pytest fixture for setting env vars and patching ``OpenAI``.
    """
    captured: dict[str, object] = {}

    def fake_openai(**kwargs: object) -> MagicMock:
        """Capture the kwargs passed to the OpenAI constructor for inspection.

        Returns:
            MagicMock: Dummy client instance to satisfy the Translator constructor.
        """
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setenv("OPENAI_TIMEOUT", "5")
    monkeypatch.setattr("translator.engine.OpenAI", fake_openai)
    Translator()
    assert captured["timeout"] == 5.0


def test_create_client_default_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaults the OpenAI client timeout to 60 seconds when ``OPENAI_TIMEOUT`` is unset.

    Args:
        monkeypatch: Pytest fixture for removing env vars and patching ``OpenAI``.
    """
    captured: dict[str, object] = {}

    def fake_openai(**kwargs: object) -> MagicMock:
        """Capture the kwargs passed to the OpenAI constructor for inspection.

        Returns:
            MagicMock: Dummy client instance to satisfy the Translator constructor.
        """
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.delenv("OPENAI_TIMEOUT", raising=False)
    monkeypatch.setattr("translator.engine.OpenAI", fake_openai)
    Translator()
    assert captured["timeout"] == 60.0


# ── _get_country_flag ──────────────────────────────────────────────────────────


def test_get_country_flag_known_language(translator: Translator) -> None:
    """Returns a non-empty flag emoji for a recognised language name.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
    """
    assert translator._get_country_flag("French") != ""


def test_get_country_flag_unknown_language(translator: Translator) -> None:
    """Returns an empty string for an unrecognisable language name.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
    """
    assert translator._get_country_flag("NotARealLanguage99") == ""


# ── get_language_info ──────────────────────────────────────────────────────────


def test_get_language_info_valid_code(translator: Translator) -> None:
    """Returns the correct name and a non-empty flag for a valid ISO 639-1 code.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
    """
    result = translator.get_language_info("fr")
    assert result["name"] == "French"
    assert result["flag"] != ""


def test_get_language_info_unknown_code(translator: Translator) -> None:
    """Falls back to the raw code as name and an empty flag for an unknown code.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
    """
    result = translator.get_language_info("xx")
    assert result["name"] == "xx"
    assert result["flag"] == ""


# ── detect_language ────────────────────────────────────────────────────────────


def _mock_detection_client(translator: Translator, reply: str) -> MagicMock:
    """Install a mocked OpenAI client returning ``reply`` and hand it back.

    Args:
        translator: Engine instance whose client is replaced.
        reply: Message content the mocked completions call returns.

    Returns:
        MagicMock: The installed mock client, for call inspection.
    """
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = reply
    translator.client = MagicMock()
    translator.client.chat.completions.create.return_value = mock_resp
    return translator.client


def test_detect_language_resolves_code(translator: Translator) -> None:
    """Resolves the model's ISO code reply to code, name, and flag.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
    """
    _mock_detection_client(translator, "en")
    result = translator.detect_language("The quick brown fox jumps over the lazy dog.")
    assert result["code"] == "en"
    assert result["name"] == "English"
    assert result["flag"] != ""


def test_detect_language_parses_decorated_reply(translator: Translator) -> None:
    """Extracts the code when the model decorates its reply with extra text.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
    """
    _mock_detection_client(translator, 'The language is French ("fr").')
    result = translator.detect_language("Bonjour le monde, comment allez-vous aujourd'hui?")
    assert result["code"] == "fr"
    assert result["name"] == "French"


def test_detect_language_truncates_prompt_to_prefix(translator: Translator) -> None:
    """Sends at most ``DETECTION_PREFIX_LIMIT`` characters of the text.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
    """
    client = _mock_detection_client(translator, "en")
    long_text = "word " * 1_000
    translator.detect_language(long_text)
    prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert long_text[:DETECTION_PREFIX_LIMIT] in prompt
    assert long_text[: DETECTION_PREFIX_LIMIT + 5] not in prompt


def test_detect_language_returns_empty_on_unparseable_reply(translator: Translator) -> None:
    """Returns an empty ``code``/``name``/``flag`` dict when no code is found.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
    """
    _mock_detection_client(translator, "I cannot determine the language.")
    result = translator.detect_language("???")
    assert result == {"code": "", "name": "", "flag": ""}


def test_detect_language_returns_empty_on_api_error(translator: Translator) -> None:
    """Returns an empty ``code``/``name``/``flag`` dict when the API call fails.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
    """
    translator.client = MagicMock()
    translator.client.chat.completions.create.side_effect = Exception("Connection refused")
    result = translator.detect_language("Hello")
    assert result == {"code": "", "name": "", "flag": ""}


# ── translate ──────────────────────────────────────────────────────────────────


def test_translate_empty_text_raises(translator: Translator) -> None:
    """Raises ``RuntimeError`` when the input text is empty.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
    """
    with pytest.raises(RuntimeError, match="Translation failed"):
        translator.translate("", "English", "en", "French", "fr")


def test_translate_returns_stripped_content(translator: Translator) -> None:
    """Strips leading and trailing whitespace from the model response.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
    """
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "  Bonjour le monde  "
    translator.client = MagicMock()
    translator.client.chat.completions.create.return_value = mock_resp

    result = translator.translate("Hello world", "English", "en", "French", "fr")

    assert result == "Bonjour le monde"


def test_translate_prompt_contains_lang_info(translator: Translator) -> None:
    """Prompt includes source and target language names with their ISO codes.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
    """
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "translation"
    translator.client = MagicMock()
    translator.client.chat.completions.create.return_value = mock_resp

    translator.translate("Hello", "English", "en", "French", "fr")

    prompt = translator.client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "English (en)" in prompt
    assert "French (fr)" in prompt
    assert "Hello" in prompt


def test_translate_prompt_contains_source_text(translator: Translator) -> None:
    """Prompt includes the original source text verbatim.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
    """
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "Hola"
    translator.client = MagicMock()
    translator.client.chat.completions.create.return_value = mock_resp

    translator.translate("Good morning", "English", "en", "Spanish", "es")

    prompt = translator.client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "Good morning" in prompt


def test_translate_unknown_source_uses_source_agnostic_prompt(translator: Translator) -> None:
    """Falls back to a source-agnostic prompt when the source language is unknown.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
    """
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "Hallo"
    translator.client = MagicMock()
    translator.client.chat.completions.create.return_value = mock_resp

    translator.translate("Hello", "", "", "German", "de")

    prompt = translator.client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "German (de)" in prompt
    assert "()" not in prompt
    assert "Hello" in prompt


def test_translate_passes_model_to_client(translator: Translator) -> None:
    """Passes the configured model identifier to the completions API.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
    """
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = "ok"
    translator.client = MagicMock()
    translator.client.chat.completions.create.return_value = mock_resp

    translator.translate("Hi", "English", "en", "French", "fr")

    assert translator.client.chat.completions.create.call_args.kwargs["model"] == translator.model


def test_translate_raises_on_connection_error(translator: Translator) -> None:
    """Raises ``RuntimeError`` when the API call fails with an exception.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
    """
    translator.client = MagicMock()
    translator.client.chat.completions.create.side_effect = Exception("Connection refused")

    with pytest.raises(RuntimeError, match="Translation failed"):
        translator.translate("Hello", "English", "en", "French", "fr")


def test_translate_raises_on_non_string_content(translator: Translator) -> None:
    """Raises ``RuntimeError`` when the model returns non-string content.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
    """
    mock_resp = MagicMock()
    mock_resp.choices[0].message.content = None
    translator.client = MagicMock()
    translator.client.chat.completions.create.return_value = mock_resp

    with pytest.raises(RuntimeError, match="Translation failed"):
        translator.translate("Hello", "English", "en", "French", "fr")


# ── log content ────────────────────────────────────────────────────────────────

# Stands in for input text or model output; it must never reach a log line.
_SYNTHETIC_MARKER = "zz-synthetic-marker-4817"


def _upstream_error(marker: str) -> BadRequestError:
    """Build the error the OpenAI client raises for an HTTP 400 that quotes ``marker``.

    Args:
        marker: Synthetic text standing in for request content echoed by the server.

    Returns:
        BadRequestError: Error whose message and body both contain ``marker``.
    """
    body = {"error": {"message": f"rejected input: {marker}"}}
    return BadRequestError(f"Error code: 400 - {body}", response=MagicMock(status_code=400, headers={}), body=body)


def test_detect_language_logs_fixed_reason_not_reply(translator: Translator, caplog: pytest.LogCaptureFixture) -> None:
    """An unparseable detection reply is logged as a fixed reason, never quoted.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
        caplog: Pytest fixture capturing log records.
    """
    _mock_detection_client(translator, f"Sorry, {_SYNTHETIC_MARKER} is all I can offer.")
    assert translator.detect_language("Lorem ipsum dolor sit amet.") == {"code": "", "name": "", "flag": ""}
    assert "No ISO 639-1 code found in detection response." in caplog.text
    assert _SYNTHETIC_MARKER not in caplog.text


def test_detect_language_logs_no_upstream_error_body(translator: Translator, caplog: pytest.LogCaptureFixture) -> None:
    """An upstream API error during detection is logged by type and status only.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
        caplog: Pytest fixture capturing log records.
    """
    translator.client = MagicMock()
    translator.client.chat.completions.create.side_effect = _upstream_error(_SYNTHETIC_MARKER)
    assert translator.detect_language("Lorem ipsum dolor sit amet.") == {"code": "", "name": "", "flag": ""}
    assert "BadRequestError (HTTP 400)" in caplog.text
    assert _SYNTHETIC_MARKER not in caplog.text


def test_translate_logs_no_upstream_error_body(translator: Translator, caplog: pytest.LogCaptureFixture) -> None:
    """An upstream API error during translation is logged by type and status only.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
        caplog: Pytest fixture capturing log records.
    """
    translator.client = MagicMock()
    translator.client.chat.completions.create.side_effect = _upstream_error(_SYNTHETIC_MARKER)
    with pytest.raises(RuntimeError, match="Translation failed"):
        translator.translate("Lorem ipsum dolor sit amet.", "English", "en", "French", "fr")
    assert "BadRequestError (HTTP 400)" in caplog.text
    assert _SYNTHETIC_MARKER not in caplog.text


def test_get_language_info_logs_no_requested_code(translator: Translator, caplog: pytest.LogCaptureFixture) -> None:
    """A free-form source language that cannot be resolved is not echoed into the log.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
        caplog: Pytest fixture capturing log records.
    """
    assert translator.get_language_info(_SYNTHETIC_MARKER) == {"name": _SYNTHETIC_MARKER, "flag": ""}
    assert "LookupError" in caplog.text
    assert _SYNTHETIC_MARKER not in caplog.text


# ── InferenceTrace: correlation id and reported usage ──────────────────────────


def _reply(content: str | None, *, prompt_tokens: int, completion_tokens: int, usage: bool = True) -> ChatCompletion:
    """Build a chat-completions reply as the OpenAI client returns it.

    Args:
        content: The message content, or ``None`` for a reply without text.
        prompt_tokens: The prompt tokens the endpoint reports.
        completion_tokens: The completion tokens the endpoint reports.
        usage: Whether the reply reports usage at all.

    Returns:
        ChatCompletion: The reply, naming ``zz-served-model`` as the model that served.
    """
    message = ChatCompletionMessage(role="assistant", content=content)
    reported = CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return ChatCompletion(
        id="chatcmpl-zz-synthetic",
        object="chat.completion",
        created=0,
        model="zz-served-model",
        choices=[Choice(index=0, finish_reason="stop", message=message)],
        usage=reported if usage else None,
    )


def test_trace_forwards_request_id_and_sums_usage(translator: Translator) -> None:
    """Both calls of a request carry its correlation id, and their usage adds up.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
    """
    translator.client = MagicMock()
    create = translator.client.chat.completions.create
    create.side_effect = [
        _reply("en", prompt_tokens=40, completion_tokens=1),
        _reply("Bonjour", prompt_tokens=60, completion_tokens=20),
    ]
    trace = InferenceTrace(request_id="zz-request-0001")

    translator.detect_language("Hello", trace=trace)
    translator.translate("Hello", "English", "en", "French", "fr", trace=trace)

    assert [call.kwargs["extra_headers"] for call in create.call_args_list] == [{"X-Request-ID": "zz-request-0001"}] * 2
    assert (trace.calls, trace.input_tokens, trace.output_tokens) == (2, 100, 21)
    assert trace.model_id == "zz-served-model"


@pytest.mark.parametrize("trace", [None, InferenceTrace()], ids=["no-trace", "no-request-id"])
def test_no_correlation_header_without_request_id(translator: Translator, trace: InferenceTrace | None) -> None:
    """Without a correlation id, inference calls carry no extra header.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
        trace: The trace passed to the engine, if any.
    """
    translator.client = MagicMock()
    create = translator.client.chat.completions.create
    create.return_value = _reply("en", prompt_tokens=1, completion_tokens=1)

    translator.detect_language("Hello", trace=trace)
    translator.translate("Hello", "English", "en", "French", "fr", trace=trace)

    assert [call.kwargs["extra_headers"] for call in create.call_args_list] == [None, None]


@pytest.mark.parametrize("request_id", ["zz-request-0004", None], ids=["with-request-id", "without-request-id"])
def test_trace_forwards_cache_scope_on_both_calls(translator: Translator, request_id: str | None) -> None:
    """Both calls of a request carry the caller's cache scope, beside any correlation id.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
        request_id: The correlation id of the request, if any.
    """
    translator.client = MagicMock()
    create = translator.client.chat.completions.create
    create.return_value = _reply("en", prompt_tokens=1, completion_tokens=1)
    scope = cache_scope_for("zz-alice")
    trace = InferenceTrace(request_id=request_id, cache_scope=scope)

    translator.detect_language("Hello", trace=trace)
    translator.translate("Hello", "English", "en", "French", "fr", trace=trace)

    expected = {"X-Request-ID": request_id, "X-Cache-Scope": scope} if request_id else {"X-Cache-Scope": scope}
    assert [call.kwargs["extra_headers"] for call in create.call_args_list] == [expected] * 2


def test_cache_scope_is_a_stable_per_user_pseudonym() -> None:
    """A user's scope is stable within the process, differs between users, and does not contain the name."""
    alice = cache_scope_for("zz-alice")
    assert alice == cache_scope_for("zz-alice")
    assert alice != cache_scope_for("zz-bob")
    assert re.fullmatch(r"[0-9a-f]{64}", alice)
    assert "zz-alice" not in alice


def test_trace_has_no_token_counts_when_a_call_reports_none(translator: Translator) -> None:
    """Token counts are only given when every completed call reported its usage.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
    """
    translator.client = MagicMock()
    translator.client.chat.completions.create.side_effect = [
        _reply("en", prompt_tokens=40, completion_tokens=1),
        _reply("Bonjour", prompt_tokens=0, completion_tokens=0, usage=False),
    ]
    trace = InferenceTrace(request_id="zz-request-0002")

    translator.detect_language("Hello", trace=trace)
    translator.translate("Hello", "English", "en", "French", "fr", trace=trace)

    assert trace.calls == 2
    assert (trace.input_tokens, trace.output_tokens) == (None, None)


def test_trace_counts_a_completed_call_with_unusable_content(translator: Translator) -> None:
    """A reply without text still counts as a completed call, with its usage.

    Args:
        translator: Translator instance provided by the ``translator`` fixture.
    """
    translator.client = MagicMock()
    translator.client.chat.completions.create.return_value = _reply(None, prompt_tokens=60, completion_tokens=0)
    trace = InferenceTrace()

    with pytest.raises(RuntimeError, match="Translation failed"):
        translator.translate("Hello", "English", "en", "French", "fr", trace=trace)

    assert (trace.calls, trace.input_tokens, trace.output_tokens) == (1, 60, 0)


def test_empty_trace_has_no_token_counts() -> None:
    """A trace without completed calls reports no usage, rather than zero."""
    trace = InferenceTrace(request_id="zz-request-0003")
    assert (trace.calls, trace.input_tokens, trace.output_tokens, trace.model_id) == (0, None, None, None)
