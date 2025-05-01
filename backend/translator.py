import logging
from typing import Any

import flag
import pycountry
import torch
from langcodes import Language
from langdetect import detect
from nltk import sent_tokenize
from pyarabic.araby import sentence_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class Translator:
    def __init__(self):
        """
        Initializes the Translator:
        - Prepares logger for diagnostics.
        - Loads the model, tokenizer, and device.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.tokenizer, self.model, self.device = self._load_model()
        self.src_lang = None

    def _load_model(
        self, model_name: str = "google/madlad400-3b-mt", local_only: bool = True
    ) -> tuple[Any, Any, torch.device] | None:
        """
        Loads the translation model and tokenizer.

        Tries to load from local cache first. If not found, downloads from Hugging Face Hub.

        Args:
            model_name (str): The name of the pretrained model.
            local_only (bool): Whether to restrict loading to local files only.

        Returns:
            tuple: (tokenizer, model, device) if successful.
        """
        try:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
            torch_dtype = (
                torch.float16 if device.type in ["cuda", "mps"] else torch.float32
            )

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, local_files_only=True
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, torch_dtype=torch_dtype, local_files_only=local_only
                ).to(device)
                self.logger.info("✅ Loaded model from local cache.")
            except FileNotFoundError:
                self.logger.info(
                    "⬇️ Model not in local cache — downloading from Hugging Face..."
                )
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, torch_dtype=torch_dtype
                ).to(device)

            return tokenizer, model, device
        except Exception as e:
            self.logger.error(f"Error while loading model: {e}")
            raise

    def _get_country_flag(self, language_name: str) -> str:
        """
        Convert a language name to the corresponding country flag.

        Args:
            language (str): Language name (e.g. "French")

        Returns:
            str: Country flag (e.g. "🇫🇷")
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
        Detect the language of a text.

        Args:
            text (str): Text to be translated.

        Returns:
            str: Detected language
        """
        try:
            self.src_lang = detect(text)
            src_lang_name = pycountry.languages.get(alpha_2=self.src_lang).name
            country_flag = self._get_country_flag(src_lang_name)
            return {"name": src_lang_name, "flag": country_flag}
        except Exception as e:
            self.logger.error(f"Error detecting language: {e}")
            return {"name": "", "flag": ""}

    def _model_inference(self, lang: str, text: str, verbose: bool = True) -> str:
        """
        Use the model for inference on a given text input.

        Args:
            lang (str): Target language code.
            text (str): Text to translate.
            verbose (bool): Set warning if the model's context window is exceeded

        Returns:
            str: The translated text.
        """
        try:
            prompt = f"<2{lang}> {text}"
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            max_model_len = 256
            input_ids = inputs.get("input_ids")
            input_len = input_ids.shape[1]
            adjusted_max_length = max_model_len

            if input_len >= max_model_len:
                if verbose:
                    print(
                        f"⚠️ Input length ({input_len} tokens) hits or exceeds max context window ({max_model_len}). Output may be truncated or degraded."
                    )
                    adjusted_max_length = input_len
                else:
                    adjusted_max_length = max_model_len

            outputs = self.model.generate(
                **inputs,
                max_length=adjusted_max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        except Exception as e:
            self.logger.error(f"Error during model inference: {e}")
            return ""

    def translate(self, trg_lang: str, text: str) -> str:
        """
        Translates text sentence-wise between any supported MADLAD languages.

        Args:
            trg_lang (str): Target language code.
            text (str): Text to translate.

        Returns:
            str: Final translated output.
        """
        try:
            if not text:
                raise ValueError("Input text cannot be empty.")
            if self.src_lang == "ar":
                sentences = sentence_tokenize(text)  # Use pyarabic for Arabic
            else:
                sentences = sent_tokenize(text)  # Use nltk for other languages
            if not sentences:
                raise ValueError("No sentences found in the input text.")
            return " ".join(
                [
                    self._model_inference(trg_lang, sentence)
                    for sentence in sentences
                    if len(sentence) > 0
                ]
            )
        except Exception as e:
            self.logger.error(f"Error during translation pipeline: {e}")
            raise RuntimeError("Translation failed") from e
