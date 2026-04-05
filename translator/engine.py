import logging
import os
from typing import Any

import flag
import pycountry
from langcodes import Language
from langdetect import detect
from openai import OpenAI


class Translator:
    """
    Translator class for handling text translation using TranslateGemma via an
    OpenAI-compatible chat/completions API endpoint.
    """

    def __init__(self):
        """
        Initializes the Translator:
        - Prepares logger for diagnostics.
        - Creates the OpenAI-compatible API client.
        - Reads model name from environment.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = self._create_client()
        self.model = os.getenv("INFERENCE_MODEL", "google/translate-gemma-2b-it")
        self.src_lang = None

    def _create_client(self) -> OpenAI:
        """
        Creates an OpenAI-compatible client from environment variables.

        Required env vars:
            INFERENCE_API_BASE_URL: Base URL of the OpenAI-compatible endpoint.

        Optional env vars:
            INFERENCE_API_KEY: API key (defaults to "dummy" for local servers).

        Returns:
            OpenAI: Configured client instance.
        """
        base_url = os.getenv("INFERENCE_API_BASE_URL")
        api_key = os.getenv("INFERENCE_API_KEY", "dummy")
        if not base_url:
            raise ValueError(
                "INFERENCE_API_BASE_URL environment variable is required."
            )
        return OpenAI(base_url=base_url, api_key=api_key)

    def _get_country_flag(self, language_name: str) -> str:
        """
        Convert a language name to the corresponding country flag emoji.

        Args:
            language_name (str): Language name (e.g. "French").

        Returns:
            str: Country flag emoji (e.g. "🇫🇷"), or empty string on failure.
        """
        try:
            lang = Language.find(language_name)
            country_code = lang.maximize().region
            return flag.flag(country_code) if country_code else ""
        except Exception as e:
            self.logger.error(f"Error converting language to country flag: {e}")
            return ""

    def detect_language(self, text: str) -> dict[str, str]:
        """
        Detect the language of a text using langdetect.

        Args:
            text (str): Text to detect the language of.

        Returns:
            dict[str, str]: A dictionary with 'name' and 'flag' of the detected language.
        """
        try:
            self.src_lang = detect(text)
            src_lang_obj = pycountry.languages.get(alpha_2=self.src_lang)
            src_lang_name = src_lang_obj.name if src_lang_obj else self.src_lang
            country_flag = self._get_country_flag(src_lang_name)
            return {"name": src_lang_name, "flag": country_flag}
        except Exception as e:
            self.logger.error(f"Error detecting language: {e}")
            return {"name": "", "flag": ""}

    def translate(
        self,
        text: str,
        src_lang_name: str,
        src_lang_code: str,
        trg_lang_name: str,
        trg_lang_code: str,
    ) -> str:
        """Translate text using the configured TranslateGemma model.

        Constructs a structured prompt following the TranslateGemma schema and
        submits it to the OpenAI-compatible chat completions endpoint.

        Args:
            text: Source text to translate.
            src_lang_name: Human-readable source language name (e.g. ``"French"``).
            src_lang_code: ISO 639-1 source language code (e.g. ``"fr"``).
            trg_lang_name: Human-readable target language name (e.g. ``"English"``).
            trg_lang_code: ISO 639-1 target language code (e.g. ``"en"``).

        Returns:
            The translated text with leading/trailing whitespace stripped.

        Raises:
            RuntimeError: If ``text`` is empty, the API call fails, or the model
                returns non-string content.
        """
        try:
            if not text:
                raise ValueError("Input text cannot be empty.")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise RuntimeError("Translation response did not contain text content.")
            return content.strip()
        except Exception as e:
            self.logger.error(f"Error during translation: {e}")
            raise RuntimeError("Translation failed") from e
