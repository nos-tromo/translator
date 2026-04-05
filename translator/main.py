import logging
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel
from translator.engine import Translator
setup_logger()

# === FastAPI Setup ===

app = FastAPI(
    title="TranslateGemma Translation API",
    description="Translate text using Google's TranslateGemma model via an OpenAI-compatible inference endpoint.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit frontend (Docker)
        "http://127.0.0.1:8501",  # Streamlit frontend alternative
        "http://localhost:8000",  # FastAPI backend (dev)
        "http://127.0.0.1:8000",  # FastAPI backend alternative (dev)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)
translator = Translator()


class TranslationRequest(BaseModel):
    """
    Validation schema for the translation request.

    Args:
        BaseModel (pydantic.BaseModel): Pydantic BaseModel for data validation.
    """

    text: str
    target_lang: str


class DetectedLanguage(BaseModel):
    """
    Validation schema for the detected language.

    Args:
        BaseModel (pydantic.BaseModel): Pydantic BaseModel for data validation.
    """

    name: str
    flag: str


class TranslationResponse(BaseModel):
    """
    Validation schema for the translation response.

    Args:
        BaseModel (pydantic.BaseModel): Pydantic BaseModel for data validation.
    """

    translation: str
    detected_language: DetectedLanguage


def _load_language_codes(
    filename: str = "language_codes.json",
) -> dict[str, str] | None:
    """
    Loads a mapping of language codes to human-readable names from a JSON file.

    Args:
        filename (str): Name of the JSON file to load (default: 'language_codes.json').

    Returns:
        dict[str, str] | None: A dictionary mapping language codes to names, or None if the file is missing.
    """
    try:
        language_path = Path(__file__).parent / filename
        with open(language_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        logging.error(f"Error loading language code file: {e}")
        raise


@app.post(
    "/translate",
    summary="Translate text",
    description="Translates input text to a target language using TranslateGemma via an OpenAI-compatible endpoint.",
    tags=["Translation"],
    response_model=TranslationResponse,
)
def translate(req: TranslationRequest) -> TranslationResponse | None:
    """
    POST endpoint for translating text between languages.

    Accepts JSON payload with text and target language code.
    Source language is auto-detected.

    Args:
        req (TranslationRequest): The translation input parameters.

    Returns:
        TranslationResponse | None: A dictionary containing the translated text and detected language info.
    """
    try:
        LANGUAGE_NAMES = _load_language_codes()
        trg_lang_name = LANGUAGE_NAMES.get(req.target_lang, req.target_lang)
        detected_lang = translator.detect_language(req.text)
        result = translator.translate(trg_lang_name, req.text)
        return TranslationResponse(
            translation=result,
            detected_language=DetectedLanguage(
                name=detected_lang.get("name", "Unknown"),
                flag=detected_lang.get("flag", "🏳️"),
            ),
        )
    except Exception as e:
        logging.error(f"Error on /translate endpoint: {e}")
        raise HTTPException(status_code=500, detail="Translation failed.")


@app.get(
    "/languages",
    summary="List supported languages",
    description="Returns a list of supported TranslateGemma language codes with human-readable names, "
    "based on the included `language_codes.json` file.",
    tags=["Metadata"],
)
def get_languages() -> list[dict[str, str]] | None:
    """
    GET endpoint for retrieving available translation languages.

    Returns:
        list[dict]: A list of objects with 'code' and 'name' for each language.
    """
    try:
        LANGUAGE_NAMES = _load_language_codes()
        if LANGUAGE_NAMES is None:
            raise HTTPException(
                status_code=500, detail="Language codes could not be loaded."
            )
        return [
            {"code": code, "name": LANGUAGE_NAMES.get(code, code)}
            for code in LANGUAGE_NAMES
        ]
    except Exception as e:
        logging.error(f"Error on /languages endpoint: {e}")
        raise HTTPException(status_code=500, detail="Failed to load language list.")
