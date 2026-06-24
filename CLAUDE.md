# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

translator is a thin translation service: a FastAPI backend that calls a
TranslateGemma-class model via any OpenAI-compatible chat/completions endpoint,
fronted by a React single-page app (Vite + `@infra/ui`, served by nginx). It
runs no models of its own — it is a typed, audited, language-aware wrapper
around an external inference endpoint.

In the nos-tromo federation it sits next to chorus, docint, and Nextext as a
fourth app: own repo, own release cycle, joins `inference-net`, no
`data-net` attachment because it has no persistent state.

## Commands

This project uses `uv` (Python 3.11+). All commands run inside this directory.

```bash
# Install backend deps
uv sync                              # production + dev deps

# Run locally (without Docker)
uv run uvicorn translator.main:app --reload     # FastAPI on :8000, docs at /docs
(cd frontend && pnpm install && pnpm dev)        # React SPA on :5173, proxies /api → :8000

# Backend tests / lint / types
uv run pytest                                   # full suite
uv run pytest tests/test_x.py::test_name        # single test
uv run pre-commit run --all-files               # ruff + pyrefly

# Frontend lint / types / tests / build (inside frontend/)
cd frontend && pnpm lint && pnpm typecheck && pnpm test && pnpm build

# Docker (preferred — matches production)
make help                            # list build-host targets
make network                         # one-time: create the external inference-net
make build                           # build backend + frontend images
make up                              # start the stack (production shape — base docker/compose.yaml only)
make up-dev                          # like 'up', but publishes backend + frontend ports on the host
make stop                            # stop containers (keep them)
make down                            # stop + remove containers (safe; no state volumes)
make logs                            # tail combined logs
make bundle                          # ship images as a versioned .tar.gz
```

The SPA is at `http://localhost:${TRANSLATOR_HOST_PORT:-8501}` (dev overlay
maps the host port to the frontend container's nginx on :80); the FastAPI docs
at `http://localhost:8000/docs` (dev overlay only — base `docker/compose.yaml`
does not publish host ports). In pure local dev (`pnpm dev`) the SPA is on
:5173 and Vite proxies `/api` to the backend.

> The React SPA lives in `frontend/` (Vite + TypeScript + `@infra/ui`), built
> to static assets and served by nginx, which reverse-proxies `/api/*` to the
> backend — so the browser is same-origin and the backend runs no CORS.

## Architecture

```
translator/                       # the importable package
  main.py        FastAPI app. Instantiates a single Translator at startup.
                 Endpoints under /api/v1: POST /translate, GET /languages,
                 GET /health.
  engine.py      Translator class. Wraps an OpenAI-compatible client to call
                 a TranslateGemma model. Uses langdetect for source detection,
                 pycountry/langcodes/emoji-country-flag for display metadata.
  log_cfg.py     Loguru setup (stderr only; container driver rotates).
  language_map.json
                 Static ISO 639-1 → human-readable name map (~50 entries),
                 mirroring Nextext's translation target-language list.
                 Loaded by both endpoints.
```

The frontend (`frontend/`, a separate Vite/React project) never imports
`engine` or any backend module — it speaks HTTP to `/api/v1` only.

**Docker shape** (`docker/`):

- `Dockerfile.backend` — multi-stage uv build, runs `uvicorn translator.main:app`.
- `Dockerfile.frontend` — two-stage build: `node:20-alpine` builds the Vite SPA,
  `nginx:1.27-alpine` serves the static assets and proxies `/api` to the backend.
- `compose.yaml` — production shape: services on `translator-net` (internal)
  + `inference-net` (external, shared); no host ports.
- `compose.override.yaml` — dev overlay that publishes 8000 (backend) and
  `${TRANSLATOR_HOST_PORT:-8501}` (frontend).

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_BASE` | Yes | — | Base URL of the OpenAI-compatible inference endpoint (e.g. `http://vllm-router:4000/v1` or `http://ollama:11434/v1`) |
| `OPENAI_API_KEY` | No | `dummy` | API key; `dummy` is fine for local servers that don't enforce auth |
| `OPENAI_TIMEOUT` | No | `60` | Per-request timeout in seconds |
| `TRANSLATE_MODEL` | No | `google/translate-gemma-2b-it` | Model identifier passed in every chat completions request |
| `DEFAULT_TARGET_LANGUAGE` | No (build) | `English` | Default target language; passed as the `VITE_DEFAULT_TARGET_LANGUAGE` build arg and baked into the SPA at image build |
| `TRANSLATOR_HOST_PORT` | No | `8501` | Dev-only host port; mapped to the frontend container's nginx on :80 |
| `INFERENCE_NETWORK` | No | `inference-net` | External Docker network name to join |
| `LOG_LEVEL` | No | `INFO` | Minimum log level emitted on stderr |
| `EXTRA_NO_PROXY` | No | — | Comma-separated hostnames appended to `NO_PROXY`; must start with `,` |

## Key Design Decisions

- **No local models.** translator never ships or loads model weights. The
  `OPENAI_API_BASE` swap is the only thing that changes between providers
  (vllm-router, Ollama, an external OpenAI-compatible endpoint).
- **Startup-time client.** `translator = Translator()` is module-scope in
  `main.py`, so `OPENAI_API_BASE` must be set before uvicorn starts —
  otherwise the import fails with a clear `ValueError`.
- **Prompt is TranslateGemma-shaped.** The instruction includes both source
  and target language names + ISO codes; the model is asked to emit only the
  translation, no commentary. See `engine.py::Translator.translate`.
- **Same-origin, no CORS.** The SPA is served by nginx, which proxies `/api/*`
  to the backend, so browser calls are always same-origin and `main.py` runs no
  CORS middleware. The backend exposes its routes under `/api/v1` (plus
  `/api/v1/health`).
- **Dev overlay publishes ports; base does not.** Production deploys front the
  frontend with the existing reverse proxy and rely on internal `expose`
  ports; the `compose.override.yaml` overlay is only used for `make up-dev` in dev.
- **No volumes at all.** The compose project declares none — logs go to
  the container logging driver — so `make down` (and even
  `docker compose down -v`) is always safe.
