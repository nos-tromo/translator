"""Translation engine backed by an OpenAI-compatible chat/completions API.

Wraps the OpenAI Python client to call a TranslateGemma model served by any
OpenAI-compatible inference backend (e.g. Ollama). Language detection is
provided by ``langdetect``; country flag emojis are resolved via ``langcodes``,
``pycountry``, and ``emoji-country-flag``.
"""

import logging
import os

import flag
import pycountry
from langcodes import Language
from langdetect import detect
from openai import OpenAI


class Translator:
    """Translation engine that calls a TranslateGemma model via a chat completions API.

    The engine detects the source language of arbitrary text, resolves human-
    readable language names and country flag emojis for display, and submits
    a structured translation prompt to the configured model.

    Attributes:
        client: OpenAI-compatible API client.
        model: Model identifier passed to every completions request.
        src_lang: ISO 639-1 code of the most recently detected source language,
            or ``None`` before the first detection call.
    """

    def __init__(self):
        """Initialise the Translator.

        Reads ``OPENAI_API_BASE`` (required) and ``TRANSLATE_MODEL`` (optional)
        from the environment and sets up the API client.

        Raises:
            ValueError: If ``OPENAI_API_BASE`` is not set.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = self._create_client()
        self.model = os.getenv("TRANSLATE_MODEL", "google/translate-gemma-2b-it")
        self.src_lang = None

    def _create_client(self) -> OpenAI:
        """Create an OpenAI-compatible client from environment variables.

        Reads the following environment variables:

        * ``OPENAI_API_BASE`` *(required)* — base URL of the inference endpoint,
          including the ``/v1`` path (e.g. ``http://ollama:11434/v1``).
        * ``OPENAI_API_KEY`` *(optional)* — API key; defaults to ``"dummy"`` for
          local servers that do not enforce authentication.

        Returns:
            OpenAI: Configured client instance.

        Raises:
            ValueError: If ``OPENAI_API_BASE`` is not set.
        """
        base_url = os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("OPENAI_API_KEY", "dummy")
        if not base_url:
            raise ValueError("OPENAI_API_BASE environment variable is required.")
        return OpenAI(base_url=base_url, api_key=api_key)

    def _get_country_flag(self, language_name: str) -> str:
        """Return the country flag emoji that best represents a language.

        Uses ``langcodes`` to maximise the language tag and extract the primary
        territory, then converts the territory code to a flag emoji.

        Args:
            language_name: Language name as a human-readable string (e.g. ``"French"``).

        Returns:
            The flag emoji (e.g. ``"🇫🇷"``), or an empty string if the language
            cannot be resolved to a territory.
        """
        try:
            lang = Language.find(language_name)
            country_code = lang.maximize().region
            return flag.flag(country_code) if country_code else ""
        except Exception as e:
            self.logger.error(f"Error converting language to country flag: {e}")
            return ""

    def get_language_info(self, code: str) -> dict[str, str]:
        """Return the display name and flag emoji for an ISO 639-1 language code.

        Args:
            code: ISO 639-1 language code (e.g. ``"fr"``).

        Returns:
            A dict with keys ``"name"`` (human-readable language name) and
            ``"flag"`` (country flag emoji). Falls back to the raw ``code`` as
            the name and an empty string as the flag if the code is not found.
        """
        try:
            lang_obj = pycountry.languages.get(alpha_2=code)
            lang_name = lang_obj.name if lang_obj else code
            country_flag = self._get_country_flag(lang_name)
            return {"name": lang_name, "flag": country_flag}
        except Exception as e:
            self.logger.error(f"Error looking up language info for '{code}': {e}")
            return {"name": code, "flag": ""}

    def detect_language(self, text: str) -> dict[str, str]:
        """Detect the language of a text string.

        Stores the detected ISO 639-1 code in ``self.src_lang`` as a side effect
        so that callers can retrieve the raw code after the call.

        Args:
            text: Text whose language should be detected.

        Returns:
            A dict with keys ``"name"`` (human-readable language name) and
            ``"flag"`` (country flag emoji). Returns ``{"name": "", "flag": ""}``
            if detection fails.
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

            prompt = (
                f"You are a professional {src_lang_name} ({src_lang_code}) to "
                f"{trg_lang_name} ({trg_lang_code}) translator. Your goal is to accurately "
                f"convey the meaning and nuances of the original {src_lang_name} text while "
                f"adhering to {trg_lang_name} grammar, vocabulary, and cultural sensitivities.\n"
                f"Produce only the {trg_lang_name} translation, without any additional "
                f"explanations or commentary. Please translate the following {src_lang_name} "
                f"text into {trg_lang_name}:\n\n\n{text}"
            )

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
