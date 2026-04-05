"""Integration tests for the FastAPI endpoints in :mod:`translator.main`.

The :class:`~translator.engine.Translator` singleton is replaced with a
:class:`unittest.mock.MagicMock` in each test that exercises ``POST /translate``
so that no real inference calls are made. ``GET /languages`` reads from the
bundled ``language_map.json`` file directly and needs no mocking under normal
conditions.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient


# ── GET /languages ─────────────────────────────────────────────────────────────

def test_get_languages_returns_list(client: TestClient) -> None:
    """``GET /languages`` responds 200 with a non-empty list.

    Args:
        client: TestClient provided by the ``client`` fixture.
    """
    r = client.get("/languages")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) > 0


def test_get_languages_entry_shape(client: TestClient) -> None:
    """Each language entry contains ``"code"`` and ``"name"`` keys.

    Args:
        client: TestClient provided by the ``client`` fixture.
    """
    r = client.get("/languages")
    for entry in r.json():
        assert "code" in entry
        assert "name" in entry


def test_get_languages_includes_english(client: TestClient) -> None:
    """English (``"en"``) is present in the language list.

    Args:
        client: TestClient provided by the ``client`` fixture.
    """
    r = client.get("/languages")
    codes = [lang["code"] for lang in r.json()]
    assert "en" in codes


def test_get_languages_returns_500_on_missing_file(client: TestClient) -> None:
    """Returns 500 when the language map file cannot be found.

    Args:
        client: TestClient provided by the ``client`` fixture.
    """
    with patch("translator.main._load_language_codes", side_effect=FileNotFoundError):
        r = client.get("/languages")
    assert r.status_code == 500


# ── POST /translate ────────────────────────────────────────────────────────────

def test_translate_auto_detect(client: TestClient) -> None:
    """``POST /translate`` auto-detects the source language and returns the translation.

    Args:
        client: TestClient provided by the ``client`` fixture.
    """
    with patch("translator.main.translator") as mock_t:
        mock_t.detect_language.return_value = {"name": "English", "flag": "🇬🇧"}
        mock_t.src_lang = "en"
        mock_t.translate.return_value = "Bonjour le monde"

        r = client.post("/translate", json={"text": "Hello world", "target_lang": "fr"})

    assert r.status_code == 200
    data = r.json()
    assert data["translation"] == "Bonjour le monde"
    assert data["detected_language"]["name"] == "English"
    assert data["detected_language"]["flag"] == "🇬🇧"


def test_translate_explicit_source_lang(client: TestClient) -> None:
    """An explicit ``source_lang`` bypasses auto-detection entirely.

    Args:
        client: TestClient provided by the ``client`` fixture.
    """
    with patch("translator.main.translator") as mock_t:
        mock_t.get_language_info.return_value = {"name": "English", "flag": "🇬🇧"}
        mock_t.translate.return_value = "Hola mundo"

        r = client.post(
            "/translate",
            json={"text": "Hello world", "target_lang": "es", "source_lang": "en"},
        )

    assert r.status_code == 200
    assert r.json()["translation"] == "Hola mundo"
    # detect_language must NOT be called when source_lang is explicit
    mock_t.detect_language.assert_not_called()


def test_translate_calls_engine_with_correct_args(client: TestClient) -> None:
    """Engine ``translate`` is called with the resolved names, codes, and text.

    Args:
        client: TestClient provided by the ``client`` fixture.
    """
    with patch("translator.main.translator") as mock_t:
        mock_t.detect_language.return_value = {"name": "English", "flag": "🇬🇧"}
        mock_t.src_lang = "en"
        mock_t.translate.return_value = "Bonjour"

        client.post("/translate", json={"text": "Hello", "target_lang": "fr"})

    mock_t.translate.assert_called_once_with("Hello", "English", "en", "French", "fr")


def test_translate_returns_500_on_engine_error(client: TestClient) -> None:
    """Returns 500 when the translation engine raises a ``RuntimeError``.

    Args:
        client: TestClient provided by the ``client`` fixture.
    """
    with patch("translator.main.translator") as mock_t:
        mock_t.detect_language.return_value = {"name": "English", "flag": ""}
        mock_t.src_lang = "en"
        mock_t.translate.side_effect = RuntimeError("Translation failed")

        r = client.post("/translate", json={"text": "Hello", "target_lang": "fr"})

    assert r.status_code == 500


def test_translate_missing_text_returns_422(client: TestClient) -> None:
    """Returns 422 when ``text`` is absent from the request body.

    Args:
        client: TestClient provided by the ``client`` fixture.
    """
    r = client.post("/translate", json={"target_lang": "fr"})
    assert r.status_code == 422


def test_translate_missing_target_lang_returns_422(client: TestClient) -> None:
    """Returns 422 when ``target_lang`` is absent from the request body.

    Args:
        client: TestClient provided by the ``client`` fixture.
    """
    r = client.post("/translate", json={"text": "Hello"})
    assert r.status_code == 422


def test_translate_unknown_target_lang_falls_back_to_code(client: TestClient) -> None:
    """Unknown target language codes are passed through to the engine as-is.

    Args:
        client: TestClient provided by the ``client`` fixture.
    """
    with patch("translator.main.translator") as mock_t:
        mock_t.detect_language.return_value = {"name": "English", "flag": ""}
        mock_t.src_lang = "en"
        mock_t.translate.return_value = "some translation"

        r = client.post("/translate", json={"text": "Hello", "target_lang": "xx"})

    assert r.status_code == 200
    # target lang name falls back to the raw code "xx"
    mock_t.translate.assert_called_once()
    assert mock_t.translate.call_args.args[3] == "xx"
