"""Unit tests for :class:`translator.engine.Translator`.

Each public method and the constructor are covered in isolation. The OpenAI
client is replaced with a :class:`unittest.mock.MagicMock` wherever a network
call would otherwise be made.
"""

from unittest.mock import MagicMock

import pytest

from translator.engine import DETECTION_PREFIX_LIMIT, Translator


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
