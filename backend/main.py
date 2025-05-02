import logging
from datetime import datetime
from pathlib import Path

# === Logging Configuration ===

logfile_name = f"backend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logs_dir = Path(".logs")
logs_dir.mkdir(parents=True, exist_ok=True)
logfile_path = logs_dir / logfile_name

file_handler = logging.FileHandler(logfile_path, encoding="utf-8")
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)

stream_handler = logging.StreamHandler()
stream_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
stream_handler.setFormatter(stream_formatter)

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, stream_handler],
)


import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from translator import Translator

# === FastAPI Setup ===

app = FastAPI(
    title="MADLAD-400 Translation API",
    description="Translate text using Google's MADLAD-400 model via Hugging Face Transformers.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",  # Docker container
        "http://127.0.0.1:8080",  # Docker container alternative
        "http://localhost:5173",  # Vite frontend (dev)
        "http://127.0.0.1:8000",  # FastAPI backend (dev)
        "http://localhost:8000",  # FastAPI backend alternative (dev)
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
    description="Translates input text to a target language using the MADLAD-400 model.",
    tags=["Translation"],
    response_model=TranslationResponse,
)
def translate(req: TranslationRequest) -> TranslationResponse | None:
    """
    POST endpoint for translating text between languages.

    Accepts JSON payload with text and target language.
    If source language is 'auto', language detection will be used.

    Args:
        req (TranslationRequest): The translation input parameters.

    Returns:
        TranslationResponse | None: A dictionary containing the translated text and detected language info.
    """
    try:
        detected_lang = translator.detect_language(req.text)
        result = translator.translate(req.target_lang, req.text)
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
    description="Returns a list of supported MADLAD-400 language codes with human-readable names, "
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
