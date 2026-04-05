# Translator

A self-hosted translation service powered by [TranslateGemma](https://huggingface.co/google/translate-gemma-2b-it) via an OpenAI-compatible inference backend (e.g. [Ollama](https://ollama.com)).

## Architecture

| Component | File | Description |
|---|---|---|
| Backend | `translator/main.py` | FastAPI app — `POST /translate`, `GET /languages` |
| Engine | `translator/engine.py` | `Translator` class — language detection, flag lookup, LLM call |
| Frontend | `translator/app.py` | Streamlit UI for text and file translation |
| Language map | `translator/language_map.json` | ~100 language codes → human-readable names |

## Prerequisites

An OpenAI-compatible inference server with TranslateGemma loaded. With Ollama:

```bash
# Create Docker network and persistent cache
docker network create inference-net
docker volume create ollama-cache

# Run the Ollama service
docker run -d \
  --network inference-net \
  --name ollama \
  -v ollama-cache:/root/.ollama \
  -p 11434:11434 \
  ollama/ollama

# Preload translation model
docker exec ollama ollama pull translategemma:4b
```

## Local development

```bash
uv sync --group dev

# Backend (FastAPI)
OPENAI_API_BASE=http://localhost:11434/v1 uv run uvicorn translator.main:app --reload
# API docs → http://127.0.0.1:8000/docs

# Frontend (Streamlit)
uv run streamlit run translator/app.py
# UI → http://localhost:8501
```

## Docker

```bash
docker compose up --build
```

App → `http://localhost:8501`

The compose file expects an external Docker network named `inference-net` (configurable via `INFERENCE_NETWORK`) with an Ollama container reachable as `ollama`.

## Environment variables

Copy `.env.example` to `.env` and fill in your values.

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_BASE` | Yes | — | Base URL of the OpenAI-compatible endpoint — must include `/v1` (e.g. `http://localhost:11434/v1`) |
| `OPENAI_API_KEY` | No | `ollama` | API key (`ollama` works for local Ollama servers) |
| `TRANSLATE_MODEL` | No | `translategemma:4b` | Model identifier |
| `BACKEND_URL` | No | `http://localhost:8000` | Frontend → backend URL |

## Tests

```bash
uv run pytest
```
