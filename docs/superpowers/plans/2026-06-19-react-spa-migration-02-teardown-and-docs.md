# React SPA Migration — Plan 02: Streamlit Teardown & Docs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the Streamlit frontend and its Python dependencies, and bring the repo's docs in line with the React SPA.

**Architecture:** Pure removal + documentation. After Plan 01 the React SPA is the live frontend; this plan removes the now-orphaned `translator/app.py` and the `streamlit`/`requests`/`types-requests` deps (used only by `app.py`), refreshes `uv.lock`, and rewrites README/CLAUDE.md/.env.example.

**Tech Stack:** `uv` (Python deps), Markdown.

## Global Constraints

- Run `git`/`uv`/`make` from the repo root.
- **Prerequisite:** Plan 01 is fully merged/complete (the React SPA is the wired frontend). Do not start otherwise — this plan deletes the Streamlit UI.
- Backend imports none of `streamlit`/`requests`/`types-requests` (verified: used only by `translator/app.py`).
- Conventional Commits; commit per task.

---

### Task 1: Delete Streamlit app and prune Python deps

**Files:**
- Delete: `translator/app.py`
- Modify: `pyproject.toml:11-25` (base deps), `:37-40` (frontend group)
- Regenerate: `uv.lock`

- [ ] **Step 1: Delete the Streamlit app**

Run: `git rm translator/app.py`
Expected: `app.py` staged for deletion.

- [ ] **Step 2: Remove the three Streamlit-only base dependencies in `pyproject.toml`**

Delete these three lines from the `[project].dependencies` array (lines 21–23):

```toml
    "requests>=2.34.2",
    "streamlit>=1.58.0",
    "types-requests>=2.33.0.20260518",
```

The remaining `dependencies` array must read exactly:

```toml
dependencies = [
    "dotenv>=0.9.9",
    "emoji-country-flag>=2.1.0",
    "fastapi>=0.137.1",
    "langcodes[data]>=3.5.1",
    "langdetect>=1.0.9",
    "loguru>=0.7.3",
    "openai>=2.43.0",
    "pycountry>=26.2.16",
    "pydantic>=2.13.4",
    "uvicorn>=0.49.0",
]
```

- [ ] **Step 3: Remove the entire `frontend` dependency group in `pyproject.toml`**

Delete this block (lines 37–40):

```toml
frontend = [
    "requests>=2.34.2",
    "streamlit>=1.58.0",
]
```

The `[dependency-groups]` section must retain only the `dev` group.

- [ ] **Step 4: Refresh the lockfile**

Run: `uv lock`
Expected: `uv.lock` updates; `streamlit`, `requests`, `types-requests` (and their now-orphaned transitive deps) are dropped.

- [ ] **Step 5: Verify the backend is intact and nothing referenced the removed deps**

Run: `grep -rnE "^[[:space:]]*(import|from)[[:space:]]+(streamlit|requests)\b" translator || echo "NO_STREAMLIT_OR_REQUESTS_IMPORTS"`
Expected: `NO_STREAMLIT_OR_REQUESTS_IMPORTS` (matches import statements only — the real failure mode — so prose containing the word "request(s)" doesn't trip it).

Run: `OPENAI_API_BASE=http://test-host:11434/v1 uv run pytest -q && uv run mypy translator`
Expected: full suite PASS; mypy clean.

- [ ] **Step 6: Commit**

```bash
git add translator/app.py pyproject.toml uv.lock
git commit -m "chore: remove Streamlit frontend and its python deps"
```

---

### Task 2: Update README.md

**Files:**
- Modify: `README.md` (architecture table, dev commands, Docker section, env table)

- [ ] **Step 1: Update the Architecture table Frontend row**

Replace:

```markdown
| Frontend | `translator/app.py` | Streamlit UI for text and file translation |
```

with:

```markdown
| Frontend | `frontend/` | React SPA (Vite + `@infra/ui`), served by nginx; speaks HTTP to the backend |
```

- [ ] **Step 2: Replace the Streamlit dev command in "Local development"**

Replace:

```markdown
# Frontend (Streamlit)
uv run streamlit run translator/app.py
# UI → http://localhost:8501
```

with:

```markdown
# Frontend (React SPA) — in a second terminal
cd frontend && pnpm install && pnpm dev
# UI → http://localhost:5173 (Vite dev server proxies /api to the backend on :8000)
```

- [ ] **Step 3: Update the Docker section**

Replace:

```markdown
## Docker

```bash
docker compose up --build
```

App → `http://localhost:8501`
```

with:

```markdown
## Docker

```bash
make up-dev   # builds + starts backend and the React SPA, publishing host ports
```

App → `http://localhost:${TRANSLATOR_HOST_PORT:-8501}`
```

- [ ] **Step 4: Replace the `BACKEND_URL` env-var row**

Replace:

```markdown
| `BACKEND_URL` | No | `http://localhost:8000` | Frontend → backend URL |
```

with:

```markdown
| `DEFAULT_TARGET_LANGUAGE` | No | `English` | Build-time default target language, baked into the SPA via the `VITE_DEFAULT_TARGET_LANGUAGE` build arg |
```

- [ ] **Step 5: Verify there is no remaining Streamlit reference**

Run: `grep -niE "streamlit|app\.py|BACKEND_URL" README.md || echo "README_CLEAN"`
Expected: `README_CLEAN`.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs(readme): describe the React SPA frontend"
```

---

### Task 3: Update CLAUDE.md and .env.example

**Files:**
- Modify: `CLAUDE.md` (overview, commands, note, architecture, docker shape, env table, design decisions)
- Modify: `.env.example` (clarify the build-time default language)

- [ ] **Step 1: Project Overview — replace the Streamlit sentence (`CLAUDE.md:7-10`)**

Replace:

```markdown
translator is a thin translation service: a FastAPI backend that calls a
TranslateGemma-class model via any OpenAI-compatible chat/completions endpoint,
fronted by a Streamlit UI. It runs no models of its own — it is a typed,
audited, language-aware wrapper around an external inference endpoint.
```

with:

```markdown
translator is a thin translation service: a FastAPI backend that calls a
TranslateGemma-class model via any OpenAI-compatible chat/completions endpoint,
fronted by a React single-page app (Vite + `@infra/ui`, served by nginx). It
runs no models of its own — it is a typed, audited, language-aware wrapper
around an external inference endpoint.
```

- [ ] **Step 2: Commands — replace the install + run blocks (`CLAUDE.md:21-32`)**

Replace:

```markdown
# Install deps
uv sync                              # production + dev deps
uv sync --only-group frontend        # frontend-only deps (no openai/langdetect)

# Run locally (without Docker)
uv run uvicorn translator.main:app --reload     # FastAPI on :8000, docs at /docs
uv run streamlit run translator/app.py          # Streamlit on :8501

# Tests / lint / types
uv run pytest                                   # full suite
uv run pytest tests/test_x.py::test_name        # single test
uv run pre-commit run --all-files               # ruff + mypy
```

with:

```markdown
# Install backend deps
uv sync                              # production + dev deps

# Run locally (without Docker)
uv run uvicorn translator.main:app --reload     # FastAPI on :8000, docs at /docs
(cd frontend && pnpm install && pnpm dev)        # React SPA on :5173, proxies /api → :8000

# Backend tests / lint / types
uv run pytest                                   # full suite
uv run pytest tests/test_x.py::test_name        # single test
uv run pre-commit run --all-files               # ruff + mypy

# Frontend lint / types / tests / build (inside frontend/)
cd frontend && pnpm lint && pnpm typecheck && pnpm test && pnpm build
```

- [ ] **Step 3: Replace the UI URL paragraph + stale README note (`CLAUDE.md:46-53`)**

Replace:

```markdown
The Streamlit UI is at `http://localhost:${TRANSLATOR_HOST_PORT:-8501}`; the
FastAPI docs at `http://localhost:8000/docs` (dev overlay only — base
`docker/compose.yaml` does not publish host ports).

> Note: the README.md is outdated — it references a `frontend/` Node.js app
> that no longer exists. The current frontend is Streamlit inside the
> `translator/` package; Streamlit is declared under `[dependency-groups].frontend`
> so the frontend image installs only that group.
```

with:

```markdown
The SPA is at `http://localhost:${TRANSLATOR_HOST_PORT:-8501}` (dev overlay
maps the host port to the frontend container's nginx on :80); the FastAPI docs
at `http://localhost:8000/docs` (dev overlay only — base `docker/compose.yaml`
does not publish host ports). In pure local dev (`pnpm dev`) the SPA is on
:5173 and Vite proxies `/api` to the backend.

> The React SPA lives in `frontend/` (Vite + TypeScript + `@infra/ui`), built
> to static assets and served by nginx, which reverse-proxies `/api/*` to the
> backend — so the browser is same-origin and the backend runs no CORS.
```

- [ ] **Step 4: Architecture — replace the `app.py` line and the HTTP note (`CLAUDE.md:64,72`)**

Replace:

```markdown
  app.py         Streamlit UI. Thin HTTP client; calls BACKEND_URL.
```

with:

```markdown
                 (the React SPA lives in the sibling `frontend/` directory,
                 not in this package — see below.)
```

Replace:

```markdown
The frontend never imports `engine` or any backend module — it speaks HTTP only.
```

with:

```markdown
The frontend (`frontend/`, a separate Vite/React project) never imports
`engine` or any backend module — it speaks HTTP to `/api/v1` only.
```

- [ ] **Step 5: Docker shape — replace the `Dockerfile.frontend` bullet (`CLAUDE.md:77-78`)**

Replace:

```markdown
- `Dockerfile.frontend` — single-stage uv build that installs only the
  `frontend` dependency group (no openai/langdetect), runs Streamlit.
```

with:

```markdown
- `Dockerfile.frontend` — two-stage build: `node:20-alpine` builds the Vite SPA,
  `nginx:1.27-alpine` serves the static assets and proxies `/api` to the backend.
```

- [ ] **Step 6: Environment Variables — replace the three frontend rows (`CLAUDE.md:92-94`)**

Replace:

```markdown
| `BACKEND_URL` | No (frontend) | `http://localhost:8000` | Where the Streamlit UI reaches the FastAPI backend |
| `DEFAULT_TARGET_LANGUAGE` | No (frontend) | `English` | Pre-selected target in the UI dropdown |
| `TRANSLATOR_HOST_PORT` | No | `8501` | Dev-only host port for the Streamlit UI |
```

with:

```markdown
| `DEFAULT_TARGET_LANGUAGE` | No (build) | `English` | Default target language; passed as the `VITE_DEFAULT_TARGET_LANGUAGE` build arg and baked into the SPA at image build |
| `TRANSLATOR_HOST_PORT` | No | `8501` | Dev-only host port; mapped to the frontend container's nginx on :80 |
```

- [ ] **Step 7: Key Design Decisions — replace the CORS bullet (`CLAUDE.md:110-112`)**

Replace:

```markdown
- **CORS is locked to localhost ports.** `main.py` allows only
  `localhost:8000` and `localhost:8501` (and their `127.0.0.1` equivalents).
  Same-origin browser calls only — there is no public CORS surface.
```

with:

```markdown
- **Same-origin, no CORS.** The SPA is served by nginx, which proxies `/api/*`
  to the backend, so browser calls are always same-origin and `main.py` runs no
  CORS middleware. The backend exposes its routes under `/api/v1` (plus
  `/api/v1/health`).
```

- [ ] **Step 8: Clarify the build-time default in `.env.example`**

Replace:

```bash
# DEFAULT_TARGET_LANGUAGE=English
```

with:

```bash
# DEFAULT_TARGET_LANGUAGE=English          # build-time SPA default; change requires `make build`
```

- [ ] **Step 9: Verify no stale Streamlit references remain in repo docs**

Run: `grep -rniE "streamlit|BACKEND_URL" CLAUDE.md .env.example || echo "DOCS_CLEAN"`
Expected: `DOCS_CLEAN`.

- [ ] **Step 10: Commit**

```bash
git add CLAUDE.md .env.example
git commit -m "docs(claude): document the React SPA, /api/v1, and same-origin model"
```

---

### Task 4: Flag the out-of-repo federation doc (no commit)

**Files:**
- Modify (out-of-repo, optional): `../CLAUDE.md`

> `infra/` is a local workspace, **not a git repo** — this edit is not committed anywhere. Do it so the federation overview stays accurate, or hand it to whoever owns the workspace doc.

- [ ] **Step 1: Update the translator stack descriptions in `infra/CLAUDE.md`**

In `../CLAUDE.md`, in the "eight projects" table, change the translator row's Stack from `FastAPI + Streamlit` to `FastAPI + React SPA`. In the "Conventions shared across the Python apps" section, the sentence "chorus, Nextext, and translator use a Streamlit UI; docint a React SPA" is now stale — update it to reflect that translator and Nextext use the React/`@infra/ui` SPA (chorus remains Streamlit).

- [ ] **Step 2: (No commit — `infra/` is not a repository.)**

---

## Definition of done (Plan 02)

- `translator/app.py` is gone; `pyproject.toml` has no `streamlit`/`requests`/`types-requests` and no `frontend` dependency group; `uv.lock` is refreshed.
- `OPENAI_API_BASE=... uv run pytest` passes and `uv run mypy translator` is clean.
- `grep -rniE "streamlit|BACKEND_URL"` across README.md, CLAUDE.md, .env.example returns nothing.
- README and CLAUDE.md describe the React SPA, the `/api/v1` contract, and the same-origin/no-CORS model.

## Self-review (against the design spec)

- **Spec coverage:** delete `app.py` + prune `streamlit`/`requests`/`types-requests` + relock + verify (T1, matches the spec's verified "used only by app.py"); README rewrite (T2); CLAUDE.md + .env.example (T3); parent `infra/CLAUDE.md` flagged as out-of-repo follow-up (T4). All teardown/doc items from the spec covered.
- **Placeholder scan:** none — every edit gives exact old→new text; verification greps have explicit expected output.
- **Consistency:** the removed deps (T1) match exactly what Plan 01's carry-forward and the spec's current-state inventory named; the env-var changes (T2/T3) match Plan 01's compose build-arg (`VITE_DEFAULT_TARGET_LANGUAGE`) and `:80` port mapping.
