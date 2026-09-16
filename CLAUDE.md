# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Data confidentiality — hard rule

**NEVER expose actual production or testing data in any file committed or
pushed to git.** This covers not only file contents but also metadata that
references real data: filenames, file descriptions, social-media account
names or handles, user identifications, sample records, log excerpts, and
screenshots. It applies everywhere git sees — source code, tests, fixtures,
docs, examples, configs, commit messages, and CI files. Use fully synthetic,
invented placeholders instead.

**Likewise, NEVER expose local filepaths from development machines** —
absolute paths or home directories such as `/Users/<name>/...`,
`/home/<name>/...`, or `C:\Users\...` — anywhere git sees. The only
permitted paths are relative project paths starting from the project's
root (e.g. `docker/compose.yaml`).

## Planning

For any non-trivial change (>1 file or any root-cause fix), present a plan and wait for approval BEFORE editing code. Do not start with Edit/Write on a fresh investigation.

## Project Overview

translator is a thin translation service: a FastAPI backend that calls an
instruction-tuned Gemma-class model via any OpenAI-compatible chat/completions endpoint,
fronted by a React single-page app (Vite + `@infra/ui`, served by nginx). It
runs no models of its own — it is a typed, audited, language-aware wrapper
around an external inference endpoint.

In the nos-tromo federation it sits next to chorus, docint, and Nextext as a
fourth app: own repo, own release cycle, backend on `inference-net` and
frontend on `edge-net`, no `data-net` attachment because it has no persistent
state.

## Commands

This project uses `uv` (Python 3.11). All commands run inside this directory.

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
make network                         # one-time: create the external inference-net + edge-net
make build                           # build backend + frontend images
make up                              # start the stack, detached; no build (production shape — base docker/compose.yaml)
make up-dev                          # like 'up' + host ports; detached, no build (run 'make build' first)
make dev                             # build, then up-dev (host ports)
make stop                            # stop containers (keep them)
make down                            # stop + remove containers (safe; no state volumes)
make logs                            # tail combined logs
make bundle                          # ship images as a .tar.gz built from the latest release tag (production)
make bundle-dev                      # ship images as a .tar.gz of the current working tree (dev/soak)
```

The SPA is at `http://localhost:${TRANSLATOR_FRONTEND_HOST_PORT:-8501}` (dev
overlay maps the host port to the frontend container's nginx on :8080); the FastAPI docs
at `http://localhost:8000/docs` (dev overlay only — base `docker/compose.yaml`
does not publish host ports). In pure local dev (`pnpm dev`) the SPA is on
:5173 and Vite proxies `/api` to the backend.

> The React SPA lives in `frontend/` (Vite + TypeScript + `@infra/ui`), built
> to static assets and served by nginx, which reverse-proxies `/api/*` to the
> backend — so the browser is same-origin and the backend runs no CORS.

## Container hardening (deploy ADR 0001)

Both containers run non-root with read-only root filesystems: the backend as
uid `10001` (`app`, `HOME=/home/app`), the frontend on
`nginxinc/nginx-unprivileged` as uid `101` listening on **:8080** (the edge
gateway's `translator-frontend` upstream must match). Compose applies
`no-new-privileges`, `cap_drop: ALL`, `read_only: true`, and a `/tmp` tmpfs to
both services via the `x-hardened` anchor. The backend writes nothing to disk;
keep it that way — any new writable path needs an explicit tmpfs or volume and
a note here.

## Architecture

```
translator/                       # the importable package
  main.py        FastAPI app. Instantiates a single Translator at startup.
                 Endpoints under /api/v1: POST /translate, GET /languages,
                 GET /version, GET /config, GET /whoami, GET /health.
  engine.py      Translator class. Wraps an OpenAI-compatible client to call
                 the configured model for both translation and source-language
                 detection (a small second completions call on a bounded text
                 prefix). Uses pycountry/langcodes/emoji-country-flag for
                 display metadata.
  log_cfg.py     Loguru setup (stderr only; container driver rotates).
  language_map.json
                 Static ISO 639-1 → human-readable name map (~50 entries),
                 mirroring Nextext's translation target-language list.
                 Loaded by both endpoints.
```

`main.py` also wires `prometheus-fastapi-instrumentator` unconditionally at
app creation, exposing aggregate HTTP metrics (no translation content) at
`GET /metrics` (unauthenticated, excluded from the OpenAPI schema) for the
`obs-plane` federation member to scrape.

The frontend (`frontend/`, a separate Vite/React project) never imports
`engine` or any backend module — it speaks HTTP to `/api/v1` only.

**Docker shape** (`docker/`):

- `Dockerfile.backend` — multi-stage uv build on the `python:3.11-slim-trixie`
  line, runs `uvicorn translator.main:app`. That interpreter line is fixed by
  `requires-python`, so Dependabot refreshes only its digest
  (`.github/dependabot.yml` ignores minor/major bumps of `library/python`);
  moving to another Python is a deliberate migration, never an automated bump.
- `Dockerfile.frontend` — two-stage build: the current `node:*-alpine` builds the
  Vite SPA (pnpm is installed with `npm install -g pnpm@<pin>` — Node ≥25 images
  ship no Corepack; keep the pin in step with `packageManager` in
  `frontend/package.json` and `pnpm-version` in `.github/workflows/ci.yml`),
  `nginxinc/nginx-unprivileged:*-alpine` serves the static assets as uid 101
  on :8080 and proxies `/api` to the backend. Both frontend images move freely
  with Dependabot.
- `compose.yaml` — production shape: both services on `translator-net`
  (internal); the backend additionally on `inference-net` and the frontend on
  `edge-net` (both external, shared). No host ports.
- `compose.override.yaml` — dev overlay that publishes
  `${TRANSLATOR_BACKEND_HOST_PORT:-8000}` (backend) and
  `${TRANSLATOR_FRONTEND_HOST_PORT:-8501}` (frontend).

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_BASE` | Yes | — | Base URL of the OpenAI-compatible inference endpoint (e.g. `http://vllm-router:4000/v1` or `http://ollama:11434/v1`) |
| `OPENAI_API_KEY` | No | `dummy` | API key; `dummy` is fine for local servers that don't enforce auth |
| `OPENAI_TIMEOUT` | No | `60` | Per-request timeout in seconds |
| `TEXT_MODEL` | Yes | — (compose fallback: `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`) | Model identifier passed in every chat completions request (translation + language detection — use an instruction-tuned model). Never hardcoded in Python — the fallback lives only in `docker/compose.yaml` |
| `DEFAULT_TARGET_LANGUAGE` | No (build) | `English` | Default target language; passed as the `VITE_DEFAULT_TARGET_LANGUAGE` build arg and baked into the SPA at image build |
| `RESPONSE_LANGUAGE` | No | `en` | SPA interface language, `en` or `de`; UI chrome only — the translation target stays a per-request choice |
| `TRANSLATOR_FRONTEND_HOST_PORT` | No | `8501` | Dev-only host port; mapped to the frontend container's nginx on :8080 |
| `TRANSLATOR_BACKEND_HOST_PORT` | No | `8000` | Dev-only host port for the FastAPI backend |
| `INFERENCE_NETWORK` | No | `inference-net` | External Docker network the backend joins |
| `EDGE_NET` | No | `edge-net` | External Docker network the frontend joins, where the edge-plane gateway reaches it as `translator-frontend` |
| `LOG_LEVEL` | No | `INFO` | Minimum log level emitted on stderr |
| `EXTRA_NO_PROXY` | No | — | Comma-separated hostnames appended to `NO_PROXY`; must start with `,` |

## Key Design Decisions

- **No local models.** translator never ships or loads model weights. The
  `OPENAI_API_BASE` swap is the only thing that changes between providers
  (vllm-router, Ollama, an external OpenAI-compatible endpoint).
- **Startup-time client.** `translator = Translator()` is module-scope in
  `main.py`, so `OPENAI_API_BASE` and `TEXT_MODEL` must be set before uvicorn
  starts — otherwise the import fails with a clear `ValueError`.
- **The LLM does the language detection.** No local detection library:
  `engine.py::Translator.detect_language` sends the first
  `DETECTION_PREFIX_LIMIT` (500) characters to the configured model and asks
  for only the ISO 639-1 code (`temperature=0`, `max_tokens=10`), parsing the
  reply defensively. This replaced `langdetect`, which misclassified short
  inputs (e.g. "Hello" → Finnish). Requires an instruction-tuned model.
- **Translation prompt names both languages.** The instruction includes source
  and target language names + ISO codes; the model is asked to emit only the
  translation, no commentary. If detection failed (empty source name) a
  source-agnostic variant is used so a wrong source language is never asserted.
  See `engine.py::Translator.translate`.
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
- **Canonical sub-path `/translator/`.** The SPA is built with Vite `base:
  '/translator/'`; the frontend's own nginx strips that prefix so its
  root-anchored locations apply unchanged, and direct root access 302s to
  `/translator/`. The frontend joins the external `edge-net` as
  `translator-frontend` — the edge-plane gateway is the sole production
  entry point and supplies `X-Auth-User` / `X-Auth-Name`. translator has no
  principal seam (unlike chorus/docint/Nextext): the sole consumer is
  `GET /api/v1/whoami`, which echoes both headers back for the SPA's
  AppHeader. It is display-only, never an authorization gate, and stays
  unauthenticated like `/version` and `/config`.

## Git & PR Workflow

- Never commit directly to `main`; always branch (`feat/`, `fix/`) and open a PR.
- Never create a NEW PR when an existing PR for the work is open — push additional commits to that branch.
- Release order is strict: bump VERSION file -> commit -> tag. Never tag before the VERSION bump.
- Use single, non-compound shell commands for `gh` operations (no `&&` chains); if `gh pr merge` is blocked, fall back to the GitHub MCP merge tool.

## Verification

Before claiming a check is green, run the actual command and paste the output. `git ls-files` does not cover untracked files — use `pre-commit run --all-files`. After opening a PR, confirm CI actually triggered on the latest push before declaring done.

## Communication Style

Keep changes minimal and scoped. Do not add explanatory code comments for trivial or self-evident changes. Do not overwrite existing test files with Write — use Edit to append or modify tests.
