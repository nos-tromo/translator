# Translator

A self-hosted translation service: paste text, pick a target language, get the
translation back. It is a thin wrapper around an instruction-tuned LLM served
over an OpenAI-compatible endpoint — the same model performs source-language
detection, so there is no local detection library and no model weights ship
with this repo.

It is one of the four application members of the nos-tromo federation and
reaches inference over the shared `inference-net` network.

## Architecture

| Component | File | Description |
|---|---|---|
| Backend | `translator/main.py` | FastAPI app — `POST /translate`, `GET /languages` |
| Engine | `translator/engine.py` | `Translator` class — language detection, flag lookup, LLM call |
| Frontend | `frontend/` | React SPA (Vite + `@infra/ui`), served by nginx; speaks HTTP to the backend |
| Language map | `translator/language_map.json` | ~50 language codes to human-readable names |

## Prerequisites

- Docker, for the containerized setup
- Python 3.11 and `uv`, for local development
- An OpenAI-compatible inference endpoint with an instruction-tuned model
  loaded. In the federation that is the shared LiteLLM router
  (`http://vllm-router:4000/v1`); standalone, anything OpenAI-compatible works.
  See [docs/configuration.md](docs/configuration.md#inference-provider-setup).

## Quick start

```bash
cp .env.example .env    # set OPENAI_API_BASE and TEXT_MODEL
make dev                # build, then start backend + SPA detached with host ports
```

App: `http://localhost:${TRANSLATOR_FRONTEND_HOST_PORT:-8501}`

`make dev` layers the dev overlay so host ports are published. `make up` runs
the base compose file alone — the production shape, no host ports. The compose
file expects an external Docker network named `inference-net` (configurable
via `INFERENCE_NETWORK`).

## Local development

```bash
uv sync --group dev

# Backend (FastAPI)
OPENAI_API_BASE=http://localhost:11434/v1 TEXT_MODEL=<model> uv run uvicorn translator.main:app --reload
# API docs at http://127.0.0.1:8000/docs

# Frontend (React SPA) — in a second terminal
cd frontend && pnpm install && pnpm dev
# UI at http://localhost:5173 (Vite proxies /api to the backend on :8000)
```

## Configuration

Two variables are required; the rest have working defaults.

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_BASE` | Yes | Base URL of the OpenAI-compatible endpoint — must include `/v1` |
| `TEXT_MODEL` | Yes | Model identifier for translation and language detection |
| `RESPONSE_LANGUAGE` | No | SPA interface language, `en` (default) or `de` |

The full table lives in [docs/configuration.md](docs/configuration.md).

## Operating

```bash
make help    # every target with a one-line description
make test    # pytest, then the frontend's vitest suite
make verify  # the pre-push gate: pre-commit (ruff + pyrefly), pnpm lint, pnpm build
```

## Documentation

- [Configuration](docs/configuration.md) — every environment variable, the
  inference-provider setup, and the production `/translator/` sub-path.
- `CLAUDE.md` — architecture detail and the design decisions behind the
  engine, kept for contributors and coding agents.
- Design history lives in `docs/` alongside, in dated design and plan files.

## Pointers

- Inference is provided by [vllm-service](https://github.com/nos-tromo/vllm-service);
  the production entry point is [edge-plane](https://github.com/nos-tromo/edge-plane).
- Questions and bugs: <https://github.com/nos-tromo/translator/issues>
