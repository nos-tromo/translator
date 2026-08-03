"""Translation engine backed by an OpenAI-compatible chat/completions API.

Wraps the OpenAI Python client to call an instruction-tuned model (e.g.
Gemma-class) served by any OpenAI-compatible inference backend. Language
detection is delegated to the same model; country flag emojis are resolved
via ``langcodes`` and ``emoji-country-flag``.
"""

import logging
import os
import re

import flag
from langcodes import Language
from openai import OpenAI

# Language detection only needs the beginning of the text; a bounded prefix
# keeps the detection call cheap regardless of input size.
DETECTION_PREFIX_LIMIT = 500

# A parenthesised code like "(fr)" is unambiguous even inside a sentence; a
# bare two-letter token is only trusted when it is the whole reply, because
# English filler words ("is", "in", "it") collide with valid ISO 639-1 codes.
_PARENTHESISED_CODE_RE = re.compile(r"\(\s*[\"']?([a-z]{2})[\"']?\s*\)")


def _parse_iso_code(reply: str) -> str | None:
    """Extract an ISO 639-1 code from a model reply, or ``None`` if absent.

    Args:
        reply: Raw message content returned by the detection call.

    Returns:
        The lowercased two-letter code, or ``None`` when the reply cannot be
        interpreted unambiguously.
    """
    bare = reply.strip().strip("\"'.`").lower()
    if len(bare) == 2 and bare.isalpha():
        return bare
    match = _PARENTHESISED_CODE_RE.search(reply.lower())
    return match.group(1) if match else None


class Translator:
    """Translation engine that calls an LLM via a chat completions API.

    The engine detects the source language of arbitrary text, resolves human-
    readable language names and country flag emojis for display, and submits
    a structured translation prompt to the configured model.

    Attributes:
        client: OpenAI-compatible API client.
        model: Model identifier passed to every completions request.
    """

    def __init__(self) -> None:
        """Initialise the Translator.

        Reads ``OPENAI_API_BASE`` and ``TEXT_MODEL`` (both required) from the
        environment and sets up the API client. No model identifier is
        hardcoded here — the fallback default lives in ``docker/compose.yaml``.

        Raises:
            ValueError: If ``OPENAI_API_BASE`` or ``TEXT_MODEL`` is not set.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.client = self._create_client()
        model = os.getenv("TEXT_MODEL")
        if not model:
            raise ValueError("TEXT_MODEL environment variable is required.")
        self.model = model

    def _create_client(self) -> OpenAI:
        """Create an OpenAI-compatible client from environment variables.

        Reads the following environment variables:

        * ``OPENAI_API_BASE`` *(required)* — base URL of the inference endpoint,
          including the ``/v1`` path (e.g. ``http://ollama:11434/v1``).
        * ``OPENAI_API_KEY`` *(optional)* — API key; defaults to ``"dummy"`` for
          local servers that do not enforce authentication.
        * ``OPENAI_TIMEOUT`` *(optional)* — request timeout in seconds; defaults
          to ``60``. Bounds the time a worker can spend waiting on the upstream
          inference endpoint.

        Returns:
            OpenAI: Configured client instance.

        Raises:
            ValueError: If ``OPENAI_API_BASE`` is not set.
        """
        base_url = os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("OPENAI_API_KEY", "dummy")
        timeout = float(os.getenv("OPENAI_TIMEOUT", "60"))
        if not base_url:
            raise ValueError("OPENAI_API_BASE environment variable is required.")
        return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

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

    @staticmethod
    def _english_language_name(code: str) -> str:
        """Resolve an ISO 639-1 code to its English display name via ``langcodes``.

        Args:
            code: ISO 639-1 language code (e.g. ``"fr"``).

        Returns:
            The English language name (e.g. ``"French"``), or the raw ``code``
            when it is not a known language tag.
        """
        try:
            lang = Language.get(code)
            if not lang.is_valid():
                return code
            return lang.display_name("en")
        except Exception:
            return code

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
            lang_name = self._english_language_name(code)
            country_flag = self._get_country_flag(lang_name)
            return {"name": lang_name, "flag": country_flag}
        except Exception as e:
            self.logger.error(f"Error looking up language info for '{code}': {e}")
            return {"name": code, "flag": ""}

    def detect_language(self, text: str) -> dict[str, str]:
        """Detect the language of a text string via the configured model.

        Sends a bounded prefix of the text (``DETECTION_PREFIX_LIMIT`` chars)
        to the chat completions endpoint and asks for the ISO 639-1 code only.
        The reply is parsed defensively (see :func:`_parse_iso_code`) so minor
        decoration — quotes, trailing punctuation, or a parenthesised code —
        does not break detection.

        Args:
            text: Text whose language should be detected.

        Returns:
            A dict with keys ``"code"`` (ISO 639-1 language code), ``"name"``
            (human-readable language name) and ``"flag"`` (country flag emoji).
            Returns ``{"code": "", "name": "", "flag": ""}`` if detection fails.
        """
        try:
            prompt = (
                "Identify the language of the following text. Reply with only the "
                "ISO 639-1 two-letter language code (for example: en, de, fr) and "
                f"nothing else.\n\nText:\n{text[:DETECTION_PREFIX_LIMIT]}"
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=10,
            )
            content = response.choices[0].message.content
            if not isinstance(content, str):
                raise RuntimeError("Language detection response did not contain text content.")
            src_lang_code = _parse_iso_code(content)
            if src_lang_code is None:
                raise RuntimeError(f"No ISO 639-1 code found in detection response: {content!r}")
            src_lang_name = self._english_language_name(src_lang_code)
            country_flag = self._get_country_flag(src_lang_name)
            return {"code": src_lang_code, "name": src_lang_name, "flag": country_flag}
        except Exception as e:
            self.logger.error(f"Error detecting language: {e}")
            return {"code": "", "name": "", "flag": ""}

    def translate(
        self,
        text: str,
        src_lang_name: str,
        src_lang_code: str,
        trg_lang_name: str,
        trg_lang_code: str,
    ) -> str:
        """Translate text using the configured model.

        Constructs a translation prompt and submits it to the OpenAI-compatible
        chat completions endpoint. When the source language is unknown (empty
        ``src_lang_name``, e.g. after a failed detection) a source-agnostic
        prompt is used instead of asserting a wrong source language.

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

            if src_lang_name:
                prompt = (
                    f"You are a professional {src_lang_name} ({src_lang_code}) to "
                    f"{trg_lang_name} ({trg_lang_code}) translator. Your goal is to accurately "
                    f"convey the meaning and nuances of the original {src_lang_name} text while "
                    f"adhering to {trg_lang_name} grammar, vocabulary, and cultural sensitivities.\n"
                    f"Produce only the {trg_lang_name} translation, without any additional "
                    f"explanations or commentary. Please translate the following {src_lang_name} "
                    f"text into {trg_lang_name}:\n\n\n{text}"
                )
            else:
                prompt = (
                    f"You are a professional translator into {trg_lang_name} ({trg_lang_code}). "
                    f"Your goal is to accurately convey the meaning and nuances of the original "
                    f"text while adhering to {trg_lang_name} grammar, vocabulary, and cultural "
                    f"sensitivities.\nProduce only the {trg_lang_name} translation, without any "
                    f"additional explanations or commentary. Please translate the following text "
                    f"into {trg_lang_name}:\n\n\n{text}"
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
