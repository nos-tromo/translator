# Streamlit → React SPA Migration — Design

**Status:** Approved (design); pending implementation plan
**Date:** 2026-06-19
**Repo:** `translator`
**Author:** session design (brainstormed)

## Overview

Replace translator's Streamlit frontend (`translator/app.py`) with a React
single-page app that consumes the shared `@infra/ui` design system, served by
nginx — the same frontend pattern docint and Nextext already run. Nextext made
this exact Streamlit→React jump (PRs around `66a7592 feat(frontend): consume
@infra/ui design system`), so this is replication of a proven pattern, not new
ground. The Streamlit UI is deleted outright (no `legacy/` archive), matching
Nextext's `chore: delete Streamlit frontend package`.

translator is the **simplest** app to migrate: a single screen, two endpoints,
no per-user data, no streaming. It therefore drops the heaviest parts of the
sibling SPAs — **no identity/`X-Auth-User` module, no SSE, no upload streaming,
no router, no client-state store**.

## Goals

- A React + TypeScript SPA at top-level `frontend/`, built with Vite/pnpm,
  served by nginx, consuming `@infra/ui` (`theme.css`, `Button`, `Card`,
  `Banner`, `cn`).
- **Exact functional parity** with the current Streamlit UI, **plus** a modern
  hover-reveal copy button on the translation output.
- nginx reverse-proxies `/api/*` to `backend:8000` **same-origin**, so the SPA
  needs no hardcoded backend URL and the backend needs no CORS.
- Backend routes move under `/api/v1` and gain `GET /api/v1/health`, matching
  the sibling convention.
- Streamlit and its Python deps are removed; Docker/Make/docs updated.
- Airgap-first preserved: deps baked at image-build time, shipped as an image
  tarball via `make bundle`. Nothing fetched at runtime.

## Non-goals

- No UX redesign beyond the copy button (no swap-languages, no char counter,
  no history). Migration and redesign stay separate.
- No authentication / principal seam. translator owns no data to scope; its
  real auth boundary remains the reverse proxy fronting the UI (see the prior
  session conclusion). The sibling `X-Auth-User` identity header is **not**
  adopted.
- No change to `engine.py` translation logic or the stateless singleton model.
- No new runtime network dependency; `@infra/ui` is build-time only.

## Current state (the "from" side)

- `translator/app.py` — Streamlit UI: source selector (with `Auto-detect`
  default), target selector (default from `DEFAULT_TARGET_LANGUAGE`), optional
  `.txt` upload, input/output text areas, "Translate" button, detected-language
  banner (`{flag} {name}`). No copy button, no swap, no char count.
- `translator/main.py` — FastAPI: `GET /languages`, `POST /translate`,
  `CORSMiddleware` locked to `localhost:8501/8000`. Stateless `Translator`
  singleton at module scope.
- `pyproject.toml` — base `dependencies` include `streamlit`, `requests`, and
  `types-requests` (the `requests` type stub); `[dependency-groups].frontend`
  duplicates `requests` + `streamlit`. All three are used only by `app.py` —
  the backend (`main.py`, `engine.py`, `log_cfg.py`) imports none of them.
- `docker/Dockerfile.frontend` — uv image, `uv sync --only-group frontend`,
  `streamlit run translator/app.py`, `EXPOSE 8501`.
- `docker/compose.yaml` — `frontend` service on `translator-net`, `expose 8501`,
  env `BACKEND_URL`, `DEFAULT_TARGET_LANGUAGE`, `STREAMLIT_*`.
- `docker/compose.override.yaml` — `${TRANSLATOR_HOST_PORT:-8501}:8501`.

## Target architecture

```
Browser ── http://host:${TRANSLATOR_HOST_PORT} ──► frontend (nginx)
                                                     ├─ /            → SPA (index.html, static assets)
                                                     └─ /api/*       → proxy_pass http://backend:8000
backend (uvicorn, translator-net + inference-net)
   GET  /api/v1/languages   → [{code,name}]
   POST /api/v1/translate   → {translation, detected_language:{name,flag}}
   GET  /api/v1/health      → {status:"ok", model}
backend ── OpenAI-compatible HTTP ──► vllm-router (inference-net)   [unchanged]
```

The browser only ever talks to the **frontend origin**; nginx is the single
seam to the backend. This holds in dev too: Vite's dev server proxies `/api` to
the backend, so the browser is same-origin in every mode. Consequence: the
backend's `CORSMiddleware` is removed as dead weight.

### Target file layout

```
translator/
  frontend/                          # NEW Node SPA (mirrors Nextext/docint)
    package.json  pnpm-lock.yaml  index.html
    vite.config.ts  tsconfig.json  tsconfig.node.json
    tailwind.config.ts  postcss.config.js  eslint.config.js  .gitignore
    nginx/  default.conf  security-headers.conf
    src/
      main.tsx  App.tsx  vite-env.d.ts
      styles/globals.css             # @import tailwindcss + @infra/ui/theme.css + @source
      lib/cn.ts                      # export { cn } from '@infra/ui'
      api/  types.ts  client.ts  queryClient.ts  translator.ts
      hooks/  useLanguages.ts  useTranslate.ts
      components/
        LanguageSelect.tsx
        TranslatePanel.tsx           # two-pane input/output layout
        OutputField.tsx              # read-only result + hover copy button
        DetectedLanguageBanner.tsx
        FileTextInput.tsx            # .txt → text (client-side FileReader)
      test/setup.ts
  translator/
    main.py            # CHANGED: /api/v1 router + /health; remove CORS
    engine.py          # UNCHANGED
    app.py             # DELETED
    language_map.json log_cfg.py     # UNCHANGED
  docker/
    Dockerfile.frontend  # REPLACED: node build → nginx serve
    Dockerfile.backend   # UNCHANGED
    compose.yaml  compose.override.yaml   # frontend service rewired
  pyproject.toml  Makefile  README.md  CLAUDE.md   # Streamlit removed, docs updated
```

## Tech stack

React 19, TypeScript 6, Vite 8, pnpm 9, Tailwind CSS v4, TanStack Query v5,
Vitest 4 + Testing Library, node 20-alpine (build), nginx 1.27-alpine (serve).
`@infra/ui` pinned `"github:nos-tromo/infra-ui#v0.1.1"`. **Excluded** vs.
siblings: react-router, Zustand, SSE, `@fontsource` (unless `@infra/ui` needs
it), identity module.

> First implementation step inventories `@infra/ui` v0.1.1's actual exports
> (pull `nos-tromo/infra-ui`) to confirm which primitives exist; composite
> controls (selects, textarea, file input) are hand-built with Tailwind +
> design tokens, since `@infra/ui` is a thin system (Button/Card/Banner/cn),
> not a full form kit.

## Component & data-flow design

- **`api/client.ts`** — native `fetch` wrappers (`apiGet`, `apiPost`) against
  `API_BASE = '/api/v1'`, an `ApiError` carrying `status` + `detail`. **No
  identity header.**
- **`api/translator.ts`** — typed calls: `getLanguages(): Promise<Language[]>`,
  `translate(req): Promise<TranslateResponse>`.
- **`api/queryClient.ts`** — TanStack QueryClient; no retry on 4xx (docint's
  policy), `refetchOnWindowFocus: false`.
- **`hooks/useLanguages.ts`** — `useQuery(['languages'])`, `staleTime: Infinity`
  (static list).
- **`hooks/useTranslate.ts`** — `useMutation` wrapping `translate()`; exposes
  `isPending`/`error` for spinner + error banner.
- **`App.tsx`** — `QueryClientProvider` → single-screen `TranslatePanel`
  (no router).

**Translate flow:** mount → `useLanguages` populates selectors (sorted by name,
name↔code maps) → user picks source (default `Auto-detect`) and target (default
`VITE_DEFAULT_TARGET_LANGUAGE`, fallback `English`), types or drops a `.txt`
file → "Translate" (disabled while empty) fires
`POST /api/v1/translate {text, target_lang, source_lang?}` (omit `source_lang`
when `Auto-detect`), `AbortController` ~120s → on success render `translation`
in `OutputField` and, if `detected_language.name` present, the
`DetectedLanguageBanner` (`{flag} {name}`, labelled "Detected source language"
for auto, "Source language" for explicit). Errors render an `@infra/ui` Banner.

**Copy button:** `OutputField` wraps the read-only result region (a styled
`whitespace-pre-wrap` div, not a disabled textarea) in a Tailwind `group`; the
copy control is `opacity-0 group-hover:opacity-100 transition`, top-right,
clipboard icon → `navigator.clipboard.writeText(translation)`, flips to a check
for ~1.5s, with an `aria-label`.

**`.txt` upload is client-side only:** `FileTextInput` reads the file via
`FileReader` (UTF-8) into the text state and sends it in the JSON `text` field.
There is **no multipart endpoint**, so nginx needs no large-body tuning.

## Backend changes (`translator/main.py`)

1. Mount `/translate` and `/languages` under `APIRouter(prefix="/api/v1")`
   → `/api/v1/translate`, `/api/v1/languages`. Request/response models and
   `engine.py` are unchanged.
2. Add `GET /api/v1/health` → `{"status":"ok","model":<TRANSLATE_MODEL>}` for
   the compose healthcheck and an upstream sanity probe.
3. Remove `CORSMiddleware` (same-origin model makes it dead weight). Reversible
   if a dev-only allowance is later wanted.

## Docker / serving

- **`Dockerfile.frontend`** (replaced): stage 1 `node:20-alpine` — corepack
  pnpm@9, `pnpm install --frozen-lockfile`, `pnpm build`; stage 2
  `nginx:1.27-alpine` — copy `dist/` + nginx confs, `EXPOSE 80`.
  `VITE_DEFAULT_TARGET_LANGUAGE` passed as a build arg.
- **`nginx/default.conf`** — `location /api/ { proxy_pass http://backend:8000; }`,
  `location /assets/ { immutable cache }`, `location / { try_files $uri
  /index.html; }`, plus `security-headers.conf`. No SSE/upload locations.
- **`compose.yaml`** — `frontend` builds `Dockerfile.frontend`, `translator-net`
  only, `depends_on: backend`, `expose: 80`, healthcheck. Drops
  `BACKEND_URL`/`STREAMLIT_*` (backend host fixed in nginx).
  **`compose.override.yaml`** — `${TRANSLATOR_HOST_PORT:-8501}:80` (container
  now serves on 80).

## Config / env changes

| Var | Change |
|---|---|
| `VITE_DEFAULT_TARGET_LANGUAGE` | NEW, build-time arg, default `English` (replaces runtime `DEFAULT_TARGET_LANGUAGE` for the UI default) |
| `BACKEND_URL` | REMOVED (browser uses same-origin `/api`; nginx targets `backend:8000`) |
| `STREAMLIT_BROWSER_GATHER_USAGE_STATS` | REMOVED |
| `TRANSLATOR_HOST_PORT` | KEPT, now maps host → container `:80` (was `:8501`) |

Default target language becomes build-time for simplicity (matches the siblings'
`VITE_` convention); changing it requires a rebuild. Runtime injection (an
nginx-entrypoint `window.__CONFIG__`) is the documented alternative if
deploy-time flexibility is later required.

## Airgap (load-bearing invariant — unchanged)

`@infra/ui` and all node deps resolve from GitHub/npm at **image-build** time
(`pnpm install --frozen-lockfile`, pinned by `pnpm-lock.yaml`). The built image
ships as a tarball via `make bundle`, identical to docint/Nextext. The
connected build host needs GitHub access (as it already does for base images);
the airgapped runtime host loads the tarball and runs nginx + static assets.
Nothing is fetched at runtime.

## Testing

- **Frontend:** Vitest + Testing Library. `api/client.test.ts` (path building,
  `ApiError` on non-2xx, no identity header) and a `TranslatePanel`
  translate-flow test with mocked fetch (languages load → translate → output +
  detected banner; copy button writes to clipboard). `pnpm typecheck && pnpm
  lint && pnpm test && pnpm build` all green.
- **Backend:** `pytest` updated to `/api/v1/*` paths; add a `/api/v1/health`
  test; remove/adjust any CORS test. `pre-commit` (ruff + mypy) clean.
- **Smoke:** `docker compose build`; `make up-dev` serves the SPA on
  `:${TRANSLATOR_HOST_PORT:-8501}`; translate round-trips; copy button works.

## Migration sequencing

translator is small enough to avoid Nextext's coexistence dance (Nextext ran a
parallel `web` service across 4 plans because it had a large job pipeline to
port incrementally). Proposed chunking (final split decided by `writing-plans`):

1. **Foundation + UI** — scaffold `frontend/` (Vite/React/TS/Tailwind/
   `@infra/ui`), API client + hooks, the parity UI + copy button; backend
   `/api/v1` + `/health` + CORS removal; nginx; Dockerfile.frontend; compose
   rewire; frontend tests + backend test updates. End state: SPA serves and
   round-trips against the backend.
2. **Teardown + docs** — delete `translator/app.py`; remove the
   `[dependency-groups].frontend` group and drop `streamlit`, `requests`, and
   `types-requests` from base `dependencies` (confirmed used only by `app.py`);
   refresh `uv.lock`; update Makefile help, README, CLAUDE.md; and flag the
   parent `infra/CLAUDE.md` table (`translator → FastAPI + Streamlit`) as an
   out-of-repo follow-up.

The `/api/v1` route move breaks Streamlit immediately, so the backend change and
the SPA land together; Streamlit is deleted in the same branch (single PR).

## Risks & mitigations

- **`@infra/ui` exports unknown until inventoried.** Mitigation: design depends
  only on `cn` + `theme.css` + at most `Button`/`Banner`/`Card`; all composite
  controls are hand-built. First task confirms exports.
- **Airgap build needs GitHub at build time.** Same constraint the siblings
  already accept; no regression. Documented in CLAUDE.md.
- **Direct callers of old root routes** (`/translate`, `/languages`) would
  break. translator's backend is consumed only by its own UI in the federation;
  risk is low. Noted for reviewers.
- **CORS removal** could surprise a developer hitting `:8000` directly from a
  browser without the proxy. Mitigated by the Vite dev proxy (the supported dev
  path); reversible.

## Decisions resolved during brainstorming

- UI: exact parity **plus** a hover-reveal copy button on the output.
- Backend: adopt `/api/v1` + `/health` (federation consistency).
- Auth: **no** principal/`X-Auth-User` seam (translator owns no data).
- Default target language: build-time `VITE_DEFAULT_TARGET_LANGUAGE`.
- Streamlit: deleted outright, no `legacy/` archive.
