"""FastAPI application for the Translator service.

Exposes endpoints under ``/api/v1``:

* ``POST /translate`` — accepts text and a target language code, auto-detects
  (or accepts an explicit) source language, and returns the translation together
  with detected language metadata. Requires an authenticated caller
  (:func:`~translator.identity.resolve_principal`; 401 without one) and writes
  one ``ai_call`` event to the data-protection audit log
  (:mod:`translator.audit`).
* ``GET /languages`` — returns the list of supported language codes and their
  human-readable names sourced from ``language_map.json``.
* ``GET /version`` — returns the running app version.
* ``GET /config`` — returns the SPA bootstrap config (UI language).
* ``GET /whoami`` — echoes the gateway's identity headers
  (``X-Auth-User``/``X-Auth-Name``) as far as they are trusted (see
  :mod:`translator.identity`), for display only.
* ``GET /health`` — liveness probe.

A single :class:`~translator.engine.Translator` instance is created at startup;
``OPENAI_API_BASE`` and ``TEXT_MODEL`` must therefore be set in the environment
before the server starts. The audit emitter, in contrast, is created in the
lifespan, so each server process writes its own chain.
"""

import json
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Annotated, cast

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field, field_validator

from translator.audit import AuditedCall, audited, get_required_actions, open_emitter, start_chain
from translator.engine import InferenceTrace, Translator
from translator.identity import get_default_identity, get_edge_secret, resolve_identity
from translator.log_cfg import setup_logger
from translator.utils.audit_event import AiCall, sha256_text

setup_logger()

MAX_TEXT_LENGTH = 20_000

SUPPORTED_UI_LANGUAGES = ("en", "de")

#: The recipient of every translation's audit event: the inference service.
INFERENCE_RECIPIENT = "inference:translate"


def load_ui_language() -> str:
    """Resolve the SPA UI language from ``RESPONSE_LANGUAGE``.

    Uniform federation variable (same name in every nos-tromo repo). It
    controls only the UI locale — the translation *target* stays a
    per-request choice and is unaffected. Unknown values fall back to
    ``en`` so a typo cannot break bring-up; a warning is logged when an
    invalid value is supplied.
    """
    raw = os.getenv("RESPONSE_LANGUAGE")
    if raw is None:
        return "en"
    candidate = raw.strip().lower()
    if candidate not in SUPPORTED_UI_LANGUAGES:
        logger.warning(f"Unknown RESPONSE_LANGUAGE value supplied; falling back to 'en' (got: {raw!r})")
        return "en"
    return candidate


# === FastAPI Setup ===

try:
    # Single source of truth: the installed package version (derives from
    # pyproject, kept in sync with the release git tag at release time).
    _APP_VERSION = _pkg_version("translator")
except PackageNotFoundError:  # running from source without an installed dist
    _APP_VERSION = "0+unknown"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Run the process's audit chain for as long as the app serves.

    Opens the chain (``start`` plus the ``config.audit`` record of the
    operating mode) before the first request and closes it with ``shutdown``
    after the last. If the chain cannot be opened, startup fails.

    Args:
        app: The application being served.

    Yields:
        None: Control to the server while it serves.
    """
    emitter = open_emitter(_APP_VERSION)
    try:
        start_chain(app, emitter)
        yield
    finally:
        emitter.close()


app = FastAPI(
    title="Translator",
    description="Translate text via an OpenAI-compatible inference endpoint.",
    version=_APP_VERSION,
    lifespan=lifespan,
)
# Same-origin model: the SPA is served by nginx, which proxies /api to this
# backend, so requests are never cross-origin and no CORS policy is needed.
router = APIRouter(prefix="/api/v1")
translator = Translator()
# Startup check: a too-short TRANSLATOR_EDGE_SECRET fails the import here; unset logs one warning.
get_edge_secret()
# Startup notice: a set TRANSLATOR_DEFAULT_IDENTITY (dev only) logs one warning.
get_default_identity()
# Startup check: an unknown action in AUDIT_REQUIRE_PURPOSE fails the import here.
get_required_actions()

# Aggregate-only request metrics (method, path template, status, latency) for
# the obs-plane Prometheus scrape target. No request/response bodies — so no
# translation text — ever reach a metric label or value.
Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


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


class ConfigResponse(BaseModel):
    """Client bootstrap config for the SPA."""

    language: str


class WhoamiResponse(BaseModel):
    """Signed-in identity, echoed from the gateway's identity headers.

    Attributes:
        username: Value of the ``X-Auth-User`` header, or ``None`` when absent
            (dev) or not trusted.
        display_name: Value of the ``X-Auth-Name`` header (Authelia
            displayname), or ``None`` when absent or not trusted.
    """

    username: str | None
    display_name: str | None


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


def _record_translation(
    call: AuditedCall,
    outcome: str,
    criteria: dict[str, object],
    trace: InferenceTrace,
    translation: str | None = None,
) -> bool:
    """Write a translation's one audit event.

    Args:
        call: The request's audit context.
        outcome: ``ok`` or ``error``.
        criteria: The request's parameters.
        trace: What the inference endpoint reported about the request's calls.
        translation: The translation returned to the caller; only its hash is
            recorded.

    Returns:
        Whether the event was written. A failure is logged by exception type
        only.
    """
    try:
        call.record(
            outcome,
            criteria=criteria,
            recipient=INFERENCE_RECIPIENT,
            ai=AiCall(
                model_alias=translator.model,
                model_id=trace.model_id,
                output_sha256=None if translation is None else sha256_text(translation),
                input_tokens=trace.input_tokens,
                output_tokens=trace.output_tokens,
            ),
        )
    except Exception as e:
        logger.error(f"Audit event for /translate could not be written: {type(e).__name__}")
        return False
    return True


@router.post(
    "/translate",
    summary="Translate text",
    description="Translates input text to a target language via an OpenAI-compatible endpoint.",
    tags=["Translation"],
    response_model=TranslationResponse,
)
def translate(
    req: TranslationRequest,
    call: Annotated[AuditedCall, Depends(audited("ai_call", "translate"))],
) -> TranslationResponse:
    """Translate text to the requested target language.

    Source language is auto-detected from ``req.text`` unless ``req.source_lang``
    is provided. The target language name is looked up in ``language_map.json``;
    unknown codes are passed through to the model as-is.

    Every request that gets this far writes exactly one ``ai_call`` event:
    ``denied`` when a required purpose is missing, ``error`` when the
    translation fails, ``ok`` otherwise. It records the language pair and the
    input length, never the text or the translation. If the event cannot be
    written, the translation is not returned.

    Args:
        req: Validated translation request containing text, target language code,
            and an optional explicit source language code.
        call: The request's audit context: who acts, the stated purpose, the
            correlation id.

    Returns:
        A :class:`TranslationResponse` with the translated text and detected
        language metadata.

    Raises:
        HTTPException: 428 if ``ai_call`` requires a stated purpose and none
            was given; 500 if translation fails, the language map cannot be
            loaded or the audit event cannot be written.
    """
    criteria: dict[str, object] = {
        "source_lang": req.source_lang or None,
        "target_lang": req.target_lang,
        "input_chars": len(req.text),
    }
    call.require_purpose(criteria)
    trace = InferenceTrace(request_id=call.request_id)
    try:
        language_names = _load_language_codes()
        trg_lang_name = language_names.get(req.target_lang, req.target_lang)
        if req.source_lang:
            detected_lang = translator.get_language_info(req.source_lang)
            src_lang_code = req.source_lang
        else:
            detected_lang = translator.detect_language(req.text, trace=trace)
            src_lang_code = detected_lang.get("code", "")
            criteria["detected_lang"] = src_lang_code or None
        src_lang_name = detected_lang.get("name", src_lang_code)
        result = translator.translate(
            req.text, src_lang_name, src_lang_code, trg_lang_name, req.target_lang, trace=trace
        )
        response = TranslationResponse(
            translation=result,
            detected_language=DetectedLanguage(
                name=detected_lang.get("name", "Unknown"),
                flag=detected_lang.get("flag", "🏳️"),
            ),
        )
    except Exception as e:
        logger.error(f"Error on /translate endpoint: {e}")
        _record_translation(call, "error", criteria, trace)
        raise HTTPException(status_code=500, detail="Translation failed.") from e
    if not _record_translation(call, "ok", criteria, trace, response.translation):
        # Fail closed: a translation that is not on record is not returned.
        raise HTTPException(status_code=500, detail="Translation failed.")
    return response


@router.get(
    "/languages",
    summary="List supported languages",
    description="Returns a list of supported language codes with human-readable names, "
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


@router.get("/version", summary="App version", tags=["Metadata"])
def get_version() -> dict[str, str]:
    """Return the running app version (unauthenticated).

    Returns:
        A dict with a single ``"version"`` key set to the installed package
        version (or ``"0+unknown"`` when running uninstalled from source).
    """
    return {"version": _APP_VERSION}


@router.get("/config", summary="SPA bootstrap config", tags=["Metadata"])
def get_config() -> ConfigResponse:
    """Return the SPA bootstrap config (UI language). Unauthenticated, like /health."""
    return ConfigResponse(language=load_ui_language())


@router.get("/whoami", summary="Signed-in identity", tags=["Metadata"])
def get_whoami(request: Request) -> WhoamiResponse:
    """Echo the gateway's identity headers, for the SPA's AppHeader.

    translator has no principal seam (unlike chorus/docint/Nextext): this is
    purely decorative header-echo for display, not an authorization gate, so
    it stays unauthenticated like ``/version`` and ``/config`` — both fields
    are ``None`` outside the gateway (dev) rather than a 401. The headers are
    read through :func:`~translator.identity.resolve_identity`, so with
    ``TRANSLATOR_EDGE_SECRET`` set, a request without the matching
    ``X-Edge-Secret`` gets the same answer as one without identity headers.

    Args:
        request: The incoming request.

    Returns:
        A :class:`WhoamiResponse` with both trusted header values (or ``None``).
    """
    identity = resolve_identity(request)
    return WhoamiResponse(username=identity.username, display_name=identity.display_name)


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
