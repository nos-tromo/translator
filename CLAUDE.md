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
runs no models of its own — it is a typed, language-aware wrapper around an
external inference endpoint.

In the nos-tromo federation it sits next to docint and Nextext as a
third app: own repo, own release cycle, backend on `inference-net` and
frontend on `edge-net`, no `data-net` attachment because it has no persistent
state.

## Commands

This project uses `uv` (Python 3.11). All commands run inside this directory.

```bash
# Install backend deps
uv sync                              # production + dev deps

# Run locally (without Docker) — no gateway, so set TRANSLATOR_DEFAULT_IDENTITY to translate
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
both services via the `x-hardened` anchor. The backend writes nothing to disk
except the audit stream's lock file, `/tmp/audit-stream.lock`, on its `/tmp`
tmpfs (`AUDIT_OUTPUT=container`, see `docs/audit.md`); keep it that way — any
new writable path needs an explicit tmpfs or volume and a note here.

## Architecture

```
translator/                       # the importable package
  main.py        FastAPI app. Instantiates a single Translator at startup;
                 the lifespan runs the process's audit chain.
                 Endpoints under /api/v1: POST /translate, GET /languages,
                 GET /version, GET /config (UI language + require_purpose),
                 GET /whoami, GET /health.
  engine.py      Translator class. Wraps an OpenAI-compatible client to call
                 the configured model for both translation and source-language
                 detection (a small second completions call on a bounded text
                 prefix). Uses pycountry/langcodes/emoji-country-flag for
                 display metadata. InferenceTrace forwards a request's
                 X-Request-ID and the caller's X-Cache-Scope pseudonym
                 (cache_scope_for, keyed per process) on both calls and
                 collects the reported model and token usage.
  identity.py    resolve_identity(): the one place that decides whether the
                 gateway's X-Auth-User / X-Auth-Name are trusted, by checking
                 X-Edge-Secret against TRANSLATOR_EDGE_SECRET.
                 resolve_principal(): the authenticated caller of /translate
                 (trusted X-Auth-User, else the dev TRANSLATOR_DEFAULT_IDENTITY
                 while no edge secret is set, else 401).
  audit.py       Data-protection audit glue: the per-process emitter, actor
                 mapping, X-Audit-Purpose, AUDIT_REQUIRE_PURPOSE (428), the
                 audited() dependency, ENDPOINT_ACTIONS (the actions the
                 endpoints record) and the require_purpose list GET /config
                 reports. Reads AUDIT_OUTPUT and AUDIT_REQUIRE_PURPOSE
                 itself, like identity.py its variables.
  utils/         audit_event.py + audit_runtime.py: byte-identical copies of
                 the canonical files in nos-tromo/.github configs/audit/.
                 Never edit them here.
  log_cfg.py     Loguru setup (stderr only; container driver rotates), the
                 filter that drops query strings from uvicorn's access log,
                 and the move of every stdout log handler (uvicorn's access
                 log) to stderr — stdout carries audit events only.
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
  `${TRANSLATOR_FRONTEND_HOST_PORT:-8501}` (frontend), and sets
  `TRANSLATOR_DEFAULT_IDENTITY` (default `dev`) so `/translate` works without
  the gateway.

## Environment Variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_BASE` | No | — (compose: `http://vllm-router:4000/v1`) | Base URL of the OpenAI-compatible inference endpoint (e.g. `http://ollama:11434/v1`); the code raises when unset outside compose |
| `OPENAI_API_KEY` | No | `dummy` | API key; `dummy` is fine for local servers that don't enforce auth |
| `OPENAI_TIMEOUT` | No | `60` | Per-request timeout in seconds |
| `TEXT_MODEL` | No | — (compose fallback: `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`) | Model identifier passed in every chat completions request (translation + language detection — use an instruction-tuned model). Never hardcoded in Python — the fallback lives only in `docker/compose.yaml` |
| `DEFAULT_TARGET_LANGUAGE` | No (build) | `English` | Default target language; passed as the `VITE_DEFAULT_TARGET_LANGUAGE` build arg and baked into the SPA at image build |
| `RESPONSE_LANGUAGE` | No | `en` | SPA interface language, `en` or `de`; UI chrome only — the translation target stays a per-request choice |
| `TRANSLATOR_EDGE_SECRET` | No | — | Secret shared with the edge gateway, sent as `X-Edge-Secret`. Set (≥ 32 characters, else startup fails): identity headers count only on requests carrying it. Unset: trusted from any caller, one startup warning. See `docs/configuration.md` |
| `TRANSLATOR_DEFAULT_IDENTITY` | No | — (dev overlay: `dev`) | Dev-only account for requests without a trusted `X-Auth-User` (else `/translate` answers 401); ignored while `TRANSLATOR_EDGE_SECRET` is set. Unset in production |
| `AUDIT_OUTPUT` | No | `stdout` (compose: `container`) | Where the audit events go: the process's stdout, or the container's main stdout under a container-wide lock. Unknown value stops startup |
| `AUDIT_REQUIRE_PURPOSE` | No | — (compose: empty too) | Audit actions that need a stated purpose (`X-Audit-Purpose`); only `ai_call` has an effect here (428 without one, and the SPA's purpose bar shows it as required). Unknown action stops startup. See `docs/audit.md` |
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
- **Ops logs carry no request content.** No log line holds input text or
  model output: engine errors log a fixed reason or the exception type via
  `engine.py::_describe_error`, never an exception message — an upstream
  error body or a language lookup can quote what it failed on. Access logs
  hold no query strings: uvicorn's passes through
  `log_cfg.py::StripQueryStringFilter`, and the frontend nginx writes its own
  `scrubbed` format (method, path without query, status, bytes, timing,
  request id — no client IP, referer or user agent). nginx's error log is
  the exception: on a request-level error, such as an unreachable backend,
  it still quotes the full request line.
- **Canonical sub-path `/translator/`.** The SPA is built with Vite `base:
  '/translator/'`; the frontend's own nginx strips that prefix so its
  root-anchored locations apply unchanged, and direct root access 302s to
  `/translator/`. The frontend joins the external `edge-net` as
  `translator-frontend` — the edge-plane gateway is the sole production
  entry point and supplies `X-Auth-User` / `X-Auth-Name`.
- **`/translate` needs a principal; the rest stays open.** Like
  docint/Nextext, translator resolves the caller fail-closed, but only
  where it must know who acts: `POST /api/v1/translate` depends on
  `identity.py::resolve_principal` (trusted `X-Auth-User`, else the dev-only
  `TRANSLATOR_DEFAULT_IDENTITY` while no edge secret is set, else 401). This
  is a 1.7.0 behaviour change — before, `/translate` served anonymous
  callers. The dev overlay sets `TRANSLATOR_DEFAULT_IDENTITY=dev` so
  `make up-dev` translates without the gateway; the production compose file
  leaves it unset, and with an edge secret set it is ignored (one startup
  warning).
  `GET /api/v1/whoami` only echoes the trusted headers for the SPA's
  AppHeader; it is display-only, never an authorization gate, and stays
  unauthenticated like `/version` and `/config`.
- **Identity headers count only with the edge secret.** The backend is also
  on the shared `inference-net`, where any container could send `X-Auth-*`
  itself. With `TRANSLATOR_EDGE_SECRET` set, the gateway sends the same value
  in `X-Edge-Secret` (the frontend nginx passes it through), and
  `identity.py::resolve_identity` trusts the identity headers only on a
  request whose `X-Edge-Secret` matches (`hmac.compare_digest`). Otherwise it
  ignores them, as if absent, and logs method and path only — never the
  secret or a header value. Unset or blank keeps the old trust-any-caller
  behaviour plus one startup warning; a value under 32 characters fails the
  import, as a missing `TEXT_MODEL` does. Code that needs the caller's
  identity calls `resolve_identity` (or `resolve_principal` where a caller is
  required) rather than reading the headers. Rollout is gateway first, then
  the secret here.
- **Every translation is an audit event, and stdout is the audit lane.**
  Each translation request writes exactly one `ai_call` event
  (data-protection audit log, §76 BDSG reference) when it completes — none
  for a request refused with 401, 422 or 400: actor, stated purpose, request
  id, recipient
  `inference:translate`, model and token usage, the SHA-256 of the
  translation, the language pair and input length — never the text or the
  translation. Audit writes fail closed: an unrecorded translation is not
  returned (generic 500). One emitter per process, created in the lifespan,
  never at import. stdout carries nothing but these events — ops logs,
  including uvicorn's access log, go to stderr. The vendored
  `translator/utils/audit_*.py` files are never edited here. Endpoint list,
  fields and knobs: `docs/audit.md`.
- **The purpose comes from the header's purpose bar, through one wrapper.**
  `@infra/ui`'s `PurposeBar` sits in `AppShell`'s `headerActions`
  (`frontend/src/components/AppPurposeBar.tsx`, labels from the i18n
  catalogs), and every API request leaves through `send()` in
  `frontend/src/api/client.ts`, which hands it to `withPurpose` under the
  scope `translator`: the stated purpose rides along as `X-Audit-Purpose`,
  and a 428 opens the bar and resends once. Its `required` flag is
  `GET /api/v1/config`'s `require_purpose`: the actions of
  `audit.py::ENDPOINT_ACTIONS` that `AUDIT_REQUIRE_PURPOSE` lists — an
  action no endpoint records refuses nothing, so it never makes the bar
  insist. The compose default stays
  empty (translator stores nothing); an operator sets `ai_call` to require a
  purpose. Keep it one place: a new audited endpoint adds its action to
  `ENDPOINT_ACTIONS` (`audited()` refuses any other), and a request the
  browser would make without `fetch` (a download link, a media element) must
  become a `fetch` through the wrapper, or it cannot carry the purpose.

## Environment example

`.env.example` follows deploy's shared format (deploy `docs/runbooks/env-sync.md`, For member authors); check with `python3 -B ../deploy/scripts/check_env_examples.py .`.

## Git & PR Workflow

- Never commit directly to `main`; always branch (`feat/`, `fix/`) and open a PR.
- Never create a NEW PR when an existing PR for the work is open — push additional commits to that branch.
- Release order is strict: bump VERSION file -> commit -> tag. Never tag before the VERSION bump.
- Use single, non-compound shell commands for `gh` operations (no `&&` chains); if `gh pr merge` is blocked, fall back to the GitHub MCP merge tool.

## Verification

Before claiming a check is green, run the actual command and paste the output. `git ls-files` does not cover untracked files — use `pre-commit run --all-files`. After opening a PR, confirm CI actually triggered on the latest push before declaring done.

## Communication Style

Keep changes minimal and scoped. Do not add explanatory code comments for trivial or self-evident changes. Do not overwrite existing test files with Write — use Edit to append or modify tests.
