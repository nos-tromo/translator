# Translator

A self-hosted translation service powered by [TranslateGemma](https://huggingface.co/google/translate-gemma-2b-it) via an OpenAI-compatible inference backend (e.g. [Ollama](https://ollama.com)).

## Architecture

| Component | File | Description |
|---|---|---|
| Backend | `translator/main.py` | FastAPI app — `POST /translate`, `GET /languages` |
| Engine | `translator/engine.py` | `Translator` class — language detection, flag lookup, LLM call |
| Frontend | `frontend/` | React SPA (Vite + `@infra/ui`), served by nginx; speaks HTTP to the backend |
| Language map | `translator/language_map.json` | ~50 language codes → human-readable names |

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
  --gpus all \
  -v ollama-cache:/root/.ollama \
  -p 11434:11434 \
  ollama/ollama:0.20.2

# Preload translation model
docker exec ollama ollama pull translategemma:4b
```

## Local development

```bash
uv sync --group dev

# Backend (FastAPI)
OPENAI_API_BASE=http://localhost:11434/v1 uv run uvicorn translator.main:app --reload
# API docs → http://127.0.0.1:8000/docs

# Frontend (React SPA) — in a second terminal
cd frontend && pnpm install && pnpm dev
# UI → http://localhost:5173 (Vite dev server proxies /api to the backend on :8000)
```

## Docker

```bash
make up-dev   # builds + starts backend and the React SPA, publishing host ports
```

App → `http://localhost:${TRANSLATOR_HOST_PORT:-8501}`

The compose file expects an external Docker network named `inference-net` (configurable via `INFERENCE_NETWORK`) with an Ollama container reachable as `ollama`.

## Environment variables

Copy `.env.example` to `.env` and fill in your values.

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENAI_API_BASE` | Yes | — | Base URL of the OpenAI-compatible endpoint — must include `/v1` (e.g. `http://localhost:11434/v1`) |
| `OPENAI_API_KEY` | No | `ollama` | API key (`ollama` works for local Ollama servers) |
| `TRANSLATE_MODEL` | No | `translategemma:4b` | Model identifier |
| `DEFAULT_TARGET_LANGUAGE` | No | `English` | Build-time default target language, baked into the SPA via the `VITE_DEFAULT_TARGET_LANGUAGE` build arg |

## Tests

```bash
uv run pytest
```
