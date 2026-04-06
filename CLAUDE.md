# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

This project uses `uv` for dependency management (Python 3.11+).

```bash
# Install dependencies
uv sync

# Run backend (FastAPI)
uv run uvicorn translator.main:app --reload
# API docs at http://127.0.0.1:8000/docs

# Run frontend (Streamlit)
uv run streamlit run translator/app.py
# UI at http://localhost:8501

# Docker (CPU or GPU)
docker compose --profile cpu up
docker compose --profile gpu up
# App at http://localhost:8080
```

Note: The README.md contains outdated instructions referencing a `frontend/` Node.js app that no longer exists — the frontend is now Streamlit inside the `translator/` package. The Streamlit dependency is not in `pyproject.toml`; it must be installed separately (`pip install streamlit requests`) or is pulled in via Dockerfile.frontend.

## Architecture

The project is a translation service with two components:

- **`translator/main.py`** — FastAPI app. Instantiates a single `Translator` at startup. Exposes `POST /translate` (auto-detects source language, translates to target) and `GET /languages`. Loads `translator/language_map.json` on each request to map language codes to names.
- **`translator/engine.py`** — `Translator` class. Wraps an OpenAI-compatible chat completions client to call a TranslateGemma model. Uses `langdetect` for source language detection, `pycountry`/`langcodes`/`emoji-country-flag` to produce flag emojis for detected languages.
- **`translator/app.py`** — Streamlit UI. Accepts text input or file upload, calls the backend API, and displays the translation result with detected language info.
- **`translator/language_map.json`** — Static mapping of ~100 language codes to human-readable names (used by both the `/languages` endpoint and the `/translate` endpoint to look up the target language name).

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_BASE` | Yes | — | Base URL of the OpenAI-compatible inference endpoint |
| `OPENAI_API_KEY` | No | `dummy` | API key (use `dummy` for local servers) |
| `TRANSLATE_MODEL` | No | `google/translate-gemma-2b-it` | Model identifier |
| `BACKEND_URL` | No | `http://localhost:8000` | Frontend → backend URL |
| `DEFAULT_TARGET_LANGUAGE` | No | `English` | Default target language in the frontend dropdown |

## Key Design Decisions

- The `Translator` client is instantiated once at FastAPI startup (`translator = Translator()` in module scope of `main.py`), so `OPENAI_API_BASE` must be set before the server starts.
- The translate prompt follows TranslateGemma's expected format: `"Translate the following text to {language}:\n{text}"`.
- CORS is restricted to localhost:8501 and localhost:8000 only.
- Logs are written to `.logs/backend_<timestamp>.log` relative to the working directory.
