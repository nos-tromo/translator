"""FastAPI application for the Translator service.

Exposes two endpoints:

* ``POST /translate`` — accepts text and a target language code, auto-detects
  (or accepts an explicit) source language, and returns the translation together
  with detected language metadata.
* ``GET /languages`` — returns the list of supported language codes and their
  human-readable names sourced from ``language_map.json``.

A single :class:`~translator.engine.Translator` instance is created at startup;
``OPENAI_API_BASE`` must therefore be set in the environment before the server
starts.
"""

import json
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import cast

from fastapi import APIRouter, FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from translator.engine import Translator
from translator.log_cfg import setup_logger

setup_logger()

MAX_TEXT_LENGTH = 20_000

# === FastAPI Setup ===

try:
    # Single source of truth: the installed package version (derives from
    # pyproject, kept in sync with the release git tag at release time).
    _APP_VERSION = _pkg_version("translator")
except PackageNotFoundError:  # running from source without an installed dist
    _APP_VERSION = "0+unknown"

app = FastAPI(
    title="Translator",
    description="Translate text via an OpenAI-compatible inference endpoint.",
    version=_APP_VERSION,
)
# Same-origin model: the SPA is served by nginx, which proxies /api to this
# backend, so requests are never cross-origin and no CORS policy is needed.
router = APIRouter(prefix="/api/v1")
translator = Translator()


class TranslationRequest(BaseModel):
    """Request body for ``POST /translate``.

    Attributes:
        text: The text to translate. Must be 1-``MAX_TEXT_LENGTH`` characters
            after stripping leading/trailing whitespace.
        target_lang: ISO 639-1 code of the desired target language (e.g. ``"fr"``).
        source_lang: ISO 639-1 code of the source language. When omitted the
            source language is auto-detected from ``text``.
    """

    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    target_lang: str
    source_lang: str | None = None

    @field_validator("text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        """Strip leading and trailing whitespace from the text.

        Args:
            value (str): The text to be stripped.

        Returns:
            str: The stripped text.

        Raises:
            ValueError: If the resulting text is empty or contains only whitespace.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must not be empty or whitespace only")
        return stripped


class DetectedLanguage(BaseModel):
    """Display metadata for a detected or resolved language.

    Attributes:
        name: Human-readable language name (e.g. ``"French"``).
        flag: Country flag emoji representing the language (e.g. ``"🇫🇷"``).
    """

    name: str
    flag: str


class TranslationResponse(BaseModel):
    """Response body for ``POST /translate``.

    Attributes:
        translation: The translated text.
        detected_language: Name and flag of the detected (or explicit) source language.
    """

    translation: str
    detected_language: DetectedLanguage


def _load_language_codes(
    filename: str = "language_map.json",
) -> dict[str, str]:
    """Load the language code → name mapping from a JSON file.

    The file is resolved relative to this module so the path is correct whether
    the package is installed or run in-place.

    Args:
        filename: Name of the JSON file located alongside this module.
            Defaults to ``"language_map.json"``.

    Returns:
        A dict mapping ISO 639-1 codes to human-readable language names.

    Raises:
        FileNotFoundError: If ``filename`` does not exist next to this module.
    """
    try:
        language_path = Path(__file__).parent / filename
        with open(language_path, encoding="utf-8") as f:
            return cast(dict[str, str], json.load(f))
    except FileNotFoundError as e:
        logger.error(f"Error loading language code file: {e}")
        raise


@router.post(
    "/translate",
    summary="Translate text",
    description="Translates input text to a target language using TranslateGemma via an OpenAI-compatible endpoint.",
    tags=["Translation"],
    response_model=TranslationResponse,
)
def translate(req: TranslationRequest) -> TranslationResponse | None:
    """Translate text to the requested target language.

    Source language is auto-detected from ``req.text`` unless ``req.source_lang``
    is provided. The target language name is looked up in ``language_map.json``;
    unknown codes are passed through to the model as-is.

    Args:
        req: Validated translation request containing text, target language code,
            and an optional explicit source language code.

    Returns:
        A :class:`TranslationResponse` with the translated text and detected
        language metadata.

    Raises:
        HTTPException: 500 if translation fails or the language map cannot be
            loaded.
    """
    try:
        LANGUAGE_NAMES = _load_language_codes()
        trg_lang_name = LANGUAGE_NAMES.get(req.target_lang, req.target_lang)
        if req.source_lang:
            detected_lang = translator.get_language_info(req.source_lang)
            src_lang_code = req.source_lang
        else:
            detected_lang = translator.detect_language(req.text)
            src_lang_code = detected_lang.get("code", "")
        src_lang_name = detected_lang.get("name", src_lang_code)
        result = translator.translate(req.text, src_lang_name, src_lang_code, trg_lang_name, req.target_lang)
        return TranslationResponse(
            translation=result,
            detected_language=DetectedLanguage(
                name=detected_lang.get("name", "Unknown"),
                flag=detected_lang.get("flag", "🏳️"),
            ),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error on /translate endpoint: {e}")
        raise HTTPException(status_code=500, detail="Translation failed.") from e


@router.get(
    "/languages",
    summary="List supported languages",
    description="Returns a list of supported TranslateGemma language codes with human-readable names, "
    "based on the included `language_map.json` file.",
    tags=["Metadata"],
)
def get_languages() -> list[dict[str, str]]:
    """Return all supported language codes with their human-readable names.

    Returns:
        A list of dicts, each containing ``"code"`` (ISO 639-1) and ``"name"``
        (human-readable) keys.

    Raises:
        HTTPException: 500 if the language map file cannot be loaded.
    """
    try:
        LANGUAGE_NAMES = _load_language_codes()
        return [{"code": code, "name": LANGUAGE_NAMES.get(code, code)} for code in LANGUAGE_NAMES]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error on /languages endpoint: {e}")
        raise HTTPException(status_code=500, detail="Failed to load language list.") from e


@router.get(
    "/health",
    summary="Health check",
    description="Liveness probe returning the configured translation model id.",
    tags=["Metadata"],
)
def health() -> dict[str, str]:
    """Return a liveness payload with the configured model identifier.

    Returns:
        A dict with ``"status"`` fixed to ``"ok"`` and ``"model"`` set to the
        engine's configured model id.
    """
    return {"status": "ok", "model": translator.model}


app.include_router(router)
