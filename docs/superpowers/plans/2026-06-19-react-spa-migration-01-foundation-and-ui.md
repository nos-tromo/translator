# React SPA Migration — Plan 01: Foundation & UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace translator's Streamlit UI with a React SPA (consuming `@infra/ui`, served by nginx) that reaches feature parity plus a hover-reveal copy button, and move the backend to a same-origin `/api/v1` contract.

**Architecture:** A Vite-built React 19 SPA served by nginx; nginx reverse-proxies `/api/*` to `backend:8000` over the Docker network, so the browser is same-origin and the backend needs no CORS. The backend's two endpoints move under `/api/v1` and gain `/api/v1/health`. Mirrors the proven docint/Nextext SPAs; translator is the thin case — no identity header, no SSE, no router, no client-state store. `engine.py` is untouched.

**Tech Stack:** React 19, TypeScript 6, Vite 8, pnpm 9, Tailwind CSS v4, TanStack Query v5, Vitest 4 + Testing Library, node 20-alpine (build), nginx 1.27-alpine (serve), `@infra/ui` `github:nos-tromo/infra-ui#v0.1.1`.

## Global Constraints

- Python floor `>=3.11,<4`; backend uses `uv`. Frontend runs **inside `frontend/`** with pnpm.
- Backend lint/type: `ruff` (google docstrings, line-length 120) + `mypy --strict`. Every new Python function needs a google-style docstring.
- `@infra/ui` is pinned exactly `"github:nos-tromo/infra-ui#v0.1.1"`. It exports `cn`, `Button`, `Card`, `Input`, `Select`, `Badge`, `Spinner`, `Banner`, and `./theme.css`. It has **no** `Textarea` — multi-line inputs are hand-built `<textarea>` with design tokens.
- Theme tokens (from `@infra/ui/theme.css`, Tailwind v4 `@theme`): colors `background foreground muted muted-foreground border accent danger primary primary-foreground`; the only per-app knob is `--app-accent` (falls back to blue). Use token utilities (`bg-background`, `text-foreground`, `border-border`, `bg-primary`, …).
- No auth / `X-Auth-User` header anywhere (translator owns no data).
- Airgap: deps baked at image-build via `pnpm install --frozen-lockfile`; nothing fetched at runtime.
- Conventional Commits, frequent commits (one per task minimum). Run `git`/`make` from the repo root; run `cd frontend` before any pnpm command.

---

### Task 1: Backend — `/api/v1` router, `/health`, remove CORS

**Files:**
- Modify: `tests/test_api.py` (repoint paths, add health tests)
- Modify: `translator/main.py:20-51` (router, health, drop CORS), `:140-215` (decorators)

**Interfaces:**
- Produces: `GET /api/v1/languages` → `list[{code,name}]`; `POST /api/v1/translate` (body `{text, target_lang, source_lang?}`) → `{translation, detected_language:{name,flag}}`; `GET /api/v1/health` → `{status:"ok", model:str}`. These are the contracts the SPA (Tasks 5–9) and nginx (Task 10) consume.

- [ ] **Step 1: Repoint test paths and add health tests in `tests/test_api.py`**

In `tests/test_api.py`, replace every endpoint literal:
- `"/languages"` → `"/api/v1/languages"` (4 occurrences: lines 25, 38, 50, 62)
- `"/translate"` → `"/api/v1/translate"` (10 occurrences: lines 83, 102, 127, 146, 157, 167, 185, 199, 209, 220)

> Do the two replacements across the **existing** file first, then append the block below — its deliberate root-path `"/translate"` in `test_root_translate_path_is_gone` must stay unprefixed.

Then append this health test section to the end of the file:

```python
# ── GET /health ────────────────────────────────────────────────────────────────


def test_health_returns_ok(client: TestClient) -> None:
    """``GET /api/v1/health`` responds 200 with status ``ok`` and the model id.

    Args:
        client: TestClient provided by the ``client`` fixture.
    """
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["model"], str)
    assert body["model"]


def test_root_translate_path_is_gone(client: TestClient) -> None:
    """The pre-migration root path ``POST /translate`` no longer exists (404).

    Args:
        client: TestClient provided by the ``client`` fixture.
    """
    r = client.post("/translate", json={"text": "Hello", "target_lang": "fr"})
    assert r.status_code == 404
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `OPENAI_API_BASE=http://test-host:11434/v1 uv run pytest tests/test_api.py -q`
Expected: FAIL — the `/api/v1/...` calls 404 (routes still at root) and `test_health_returns_ok` 404s.

- [ ] **Step 3: Add the router + health endpoint and remove CORS in `translator/main.py`**

Change the imports block (lines 20–23) from:

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field, field_validator
```

to:

```python
from fastapi import APIRouter, FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field, field_validator
```

Replace the app + middleware block (lines 34–51) from the `app = FastAPI(...)` through `translator = Translator()` with:

```python
app = FastAPI(
    title="Translator",
    description="Translate text via an OpenAI-compatible inference endpoint.",
    version="1.0.0",
)
# Same-origin model: the SPA is served by nginx, which proxies /api to this
# backend, so requests are never cross-origin and no CORS policy is needed.
router = APIRouter(prefix="/api/v1")
translator = Translator()
```

Change the two route decorators:
- `@app.post(` (line 140) → `@router.post(`
- `@app.get(` (line 191) → `@router.get(`

Add the health endpoint immediately after the `get_languages` function (end of file), then register the router:

```python


@router.get(
    "/health",
    summary="Health check",
    description="Liveness probe returning the configured translation model id.",
    tags=["Metadata"],
)
def health() -> dict[str, str]:
    """Return a liveness payload with the configured model identifier.

    Returns:
        A dict with ``"status"`` fixed to ``"ok"`` and ``"model"`` set to the
        engine's configured model id.
    """
    return {"status": "ok", "model": translator.model}


app.include_router(router)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `OPENAI_API_BASE=http://test-host:11434/v1 uv run pytest tests/test_api.py -q`
Expected: PASS (all repointed tests + 2 new health/404 tests).

- [ ] **Step 5: Lint + type-check the backend change**

Run: `uv run ruff check translator/main.py tests/test_api.py && uv run mypy translator/main.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add translator/main.py tests/test_api.py
git commit -m "feat(api): move routes under /api/v1, add /health, drop CORS"
```

---

### Task 2: Scaffold the React app skeleton

**Files:**
- Create: `frontend/package.json`, `frontend/.gitignore`, `frontend/index.html`,
  `frontend/tsconfig.json`, `frontend/tsconfig.node.json`,
  `frontend/src/vite-env.d.ts`, `frontend/src/main.tsx`, `frontend/src/App.tsx`

> The first full `pnpm build` runs in Task 4 (after `globals.css` exists). Task 2 only lays down files.

- [ ] **Step 1: Create `frontend/package.json`**

```json
{
  "name": "translator-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "packageManager": "pnpm@9.12.0",
  "scripts": {
    "dev": "vite",
    "build": "tsc -b && vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest",
    "lint": "eslint .",
    "typecheck": "tsc -b --noEmit"
  },
  "dependencies": {
    "@fontsource/inter": "^5.1.0",
    "@infra/ui": "github:nos-tromo/infra-ui#v0.1.1",
    "@tanstack/react-query": "^5.100.14",
    "react": "^19.2.0",
    "react-dom": "^19.2.0"
  },
  "devDependencies": {
    "@eslint/js": "^9.12.0",
    "@tailwindcss/postcss": "^4.3.1",
    "@testing-library/jest-dom": "^6.5.0",
    "@testing-library/react": "^16.0.1",
    "@testing-library/user-event": "^14.5.2",
    "@types/node": "^25.0.0",
    "@types/react": "^19.2.0",
    "@types/react-dom": "^19.2.0",
    "@vitejs/plugin-react": "^6.0.2",
    "eslint": "^9.12.0",
    "eslint-plugin-react-hooks": "^7.1.1",
    "happy-dom": "^20.10.3",
    "postcss": "^8.5.15",
    "tailwindcss": "^4.3.1",
    "typescript": "^6.0.3",
    "typescript-eslint": "^8.61.0",
    "vite": "^8.0.16",
    "vitest": "^4.1.8"
  }
}
```

- [ ] **Step 2: Create `frontend/.gitignore`**

```gitignore
node_modules/
dist/
*.local
.vite/
```

- [ ] **Step 3: Create `frontend/index.html`**

```html
<!doctype html>
<html lang="en" data-theme="dark">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Translator</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

- [ ] **Step 4: Create `frontend/tsconfig.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "useDefineForClassFields": true,
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "ignoreDeprecations": "5.0",
    "baseUrl": ".",
    "paths": { "@/*": ["./src/*"] }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

- [ ] **Step 5: Create `frontend/tsconfig.node.json`**

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2023"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "isolatedModules": true,
    "noEmit": true,
    "strict": true
  },
  "include": ["vite.config.ts"]
}
```

- [ ] **Step 6: Create `frontend/src/vite-env.d.ts`** (types the one custom env var)

```ts
/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_DEFAULT_TARGET_LANGUAGE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
```

- [ ] **Step 7: Create `frontend/src/main.tsx`**

```tsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import '@fontsource/inter/400.css'
import '@fontsource/inter/500.css'
import '@fontsource/inter/600.css'
import './styles/globals.css'
import App from './App'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

- [ ] **Step 8: Create `frontend/src/App.tsx`** (placeholder; Task 9 replaces it)

```tsx
export default function App() {
  return <div>Translator</div>
}
```

- [ ] **Step 9: Commit**

```bash
git add frontend/package.json frontend/.gitignore frontend/index.html \
  frontend/tsconfig.json frontend/tsconfig.node.json \
  frontend/src/vite-env.d.ts frontend/src/main.tsx frontend/src/App.tsx
git commit -m "feat(frontend): scaffold React app skeleton"
```

---

### Task 3: Vite + Vitest config, test setup, install deps

**Files:**
- Create: `frontend/vite.config.ts`, `frontend/src/test/setup.ts`
- Generated: `frontend/pnpm-lock.yaml`

- [ ] **Step 1: Create `frontend/vite.config.ts`** (dev proxy = same-origin like nginx)

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'

const BACKEND = process.env.TRANSLATOR_BACKEND_ORIGIN ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': { target: BACKEND, changeOrigin: true },
    },
  },
  test: {
    environment: 'happy-dom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
  },
})
```

- [ ] **Step 2: Create `frontend/src/test/setup.ts`**

```ts
import '@testing-library/jest-dom/vitest'
```

- [ ] **Step 3: Install dependencies (generates the lockfile)**

Run: `cd frontend && pnpm install`
Expected: completes; creates `frontend/pnpm-lock.yaml` and `frontend/node_modules/`. `@infra/ui` resolves from the GitHub tarball.

- [ ] **Step 4: Type-check the skeleton**

Run: `cd frontend && pnpm typecheck`
Expected: PASS (CSS imports in `main.tsx` are accepted via `vite/client` ambient types even though `globals.css` arrives in Task 4).

- [ ] **Step 5: Commit**

```bash
git add frontend/vite.config.ts frontend/src/test/setup.ts frontend/pnpm-lock.yaml
git commit -m "chore(frontend): vite + vitest config and lockfile"
```

---

### Task 4: Tailwind v4, PostCSS, ESLint, design-system styles, `cn`

**Files:**
- Create: `frontend/postcss.config.js`, `frontend/eslint.config.js`,
  `frontend/src/styles/globals.css`, `frontend/src/lib/cn.ts`

> Tailwind v4 is CSS-configured — there is no `tailwind.config.ts`. Tokens come from `@infra/ui/theme.css`; the `@source` line makes Tailwind scan the compiled `@infra/ui` components (which live in `node_modules`, not scanned by default).

- [ ] **Step 1: Create `frontend/postcss.config.js`**

```js
export default {
  plugins: {
    '@tailwindcss/postcss': {},
  },
}
```

- [ ] **Step 2: Create `frontend/eslint.config.js`**

```js
import js from '@eslint/js'
import tseslint from 'typescript-eslint'
import reactHooks from 'eslint-plugin-react-hooks'

export default tseslint.config(
  { ignores: ['dist/', 'node_modules/'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ['**/*.{ts,tsx}'],
    plugins: { 'react-hooks': reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    },
  },
)
```

- [ ] **Step 3: Create `frontend/src/styles/globals.css`**

```css
@import 'tailwindcss';
@import '@infra/ui/theme.css';
@source '../../node_modules/@infra/ui/dist';

:root {
  /* translator's single design-system knob; primary derives from this. */
  --app-accent: hsl(217 91% 60%);
  color-scheme: dark;
}

html,
body,
#root {
  height: 100%;
}

body {
  margin: 0;
  background: var(--color-background);
  color: var(--color-foreground);
  font-family: var(--font-sans);
}
```

- [ ] **Step 4: Create `frontend/src/lib/cn.ts`** (re-export the design system's util)

```ts
export { cn } from '@infra/ui'
```

- [ ] **Step 5: Lint, type-check, and build**

Run: `cd frontend && pnpm lint && pnpm typecheck && pnpm build`
Expected: all PASS; `vite build` writes `frontend/dist/` (the placeholder App compiles, Tailwind resolves `@infra/ui` tokens).

- [ ] **Step 6: Commit**

```bash
git add frontend/postcss.config.js frontend/eslint.config.js \
  frontend/src/styles/globals.css frontend/src/lib/cn.ts
git commit -m "chore(frontend): tailwind v4, eslint, @infra/ui theme, cn"
```

---

### Task 5: API types, fetch client, query client

**Files:**
- Create: `frontend/src/api/types.ts`, `frontend/src/api/client.ts`,
  `frontend/src/api/queryClient.ts`
- Test: `frontend/src/api/client.test.ts`

**Interfaces:**
- Produces: `Language {code,name}`, `DetectedLanguage {name,flag}`,
  `TranslateRequest {text, target_lang, source_lang?}`,
  `TranslateResponse {translation, detected_language}` (types.ts);
  `apiGet<T>(path, signal?)`, `apiPost<T>(path, json, signal?)`, `ApiError`
  (client.ts); `queryClient` (queryClient.ts). Consumed by Task 6.
- Note: **no identity/auth header** — translator owns no data.

- [ ] **Step 1: Create `frontend/src/api/types.ts`**

```ts
// Mirrors translator/main.py TranslationRequest/TranslationResponse and the
// /languages payload. Field names match the JSON exactly.

export interface Language {
  code: string
  name: string
}

export interface DetectedLanguage {
  name: string
  flag: string
}

export interface TranslateRequest {
  text: string
  target_lang: string
  source_lang?: string
}

export interface TranslateResponse {
  translation: string
  detected_language: DetectedLanguage
}
```

- [ ] **Step 2: Write the failing test — `frontend/src/api/client.test.ts`**

```ts
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, apiGet, apiPost } from './client'

afterEach(() => vi.restoreAllMocks())

function mockFetch(status: number, body: unknown) {
  const fn = vi.fn(
    async () =>
      new Response(JSON.stringify(body), {
        status,
        headers: { 'content-type': 'application/json' },
      }),
  )
  vi.stubGlobal('fetch', fn)
  return fn
}

describe('apiGet', () => {
  it('calls the /api/v1-prefixed path with no auth header', async () => {
    const fetchFn = mockFetch(200, [{ code: 'en', name: 'English' }])
    const body = await apiGet<unknown[]>('/languages')
    expect(body).toHaveLength(1)
    const [url, init] = fetchFn.mock.calls[0]
    expect(url).toBe('/api/v1/languages')
    expect((init as RequestInit).headers ?? {}).not.toHaveProperty('X-Auth-User')
  })

  it('throws ApiError carrying status and detail on non-2xx', async () => {
    mockFetch(500, { detail: 'Translation failed.' })
    await expect(apiGet('/translate')).rejects.toMatchObject({
      name: 'ApiError',
      status: 500,
      detail: 'Translation failed.',
    } satisfies Partial<ApiError>)
  })
})

describe('apiPost', () => {
  it('serializes a JSON body and sets content-type', async () => {
    const fetchFn = mockFetch(200, { translation: 'Bonjour', detected_language: { name: 'English', flag: '🇬🇧' } })
    await apiPost('/translate', { text: 'Hello', target_lang: 'fr' })
    const [url, init] = fetchFn.mock.calls[0]
    const ri = init as RequestInit
    expect(url).toBe('/api/v1/translate')
    expect(ri.method).toBe('POST')
    expect(ri.body).toBe(JSON.stringify({ text: 'Hello', target_lang: 'fr' }))
    expect(ri.headers).toMatchObject({ 'content-type': 'application/json' })
  })
})
```

- [ ] **Step 2b: Run the test to verify it fails**

Run: `cd frontend && pnpm test -- client`
Expected: FAIL — `Cannot find module './client'`.

- [ ] **Step 3: Create `frontend/src/api/client.ts`**

```ts
export const API_BASE = '/api/v1'

export class ApiError extends Error {
  readonly status: number
  readonly detail: unknown
  constructor(status: number, detail: unknown) {
    super(`API ${status}: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
  }
}

async function parse<T>(res: Response): Promise<T> {
  const text = await res.text()
  let body: unknown = text
  try {
    body = text ? JSON.parse(text) : null
  } catch {
    /* keep raw text */
  }
  if (!res.ok) {
    const detail =
      body && typeof body === 'object' && 'detail' in body
        ? (body as { detail: unknown }).detail
        : body
    throw new ApiError(res.status, detail)
  }
  return body as T
}

export async function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { signal })
  return parse<T>(res)
}

export async function apiPost<T>(path: string, json: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(json),
    signal,
  })
  return parse<T>(res)
}
```

- [ ] **Step 4: Create `frontend/src/api/queryClient.ts`**

```ts
import { QueryClient } from '@tanstack/react-query'
import { ApiError } from './client'

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      refetchOnWindowFocus: false,
      retry: (failureCount, error) => {
        if (error instanceof ApiError && error.status >= 400 && error.status < 500) {
          return false
        }
        return failureCount < 1
      },
    },
  },
})
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd frontend && pnpm test -- client`
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add frontend/src/api/types.ts frontend/src/api/client.ts \
  frontend/src/api/queryClient.ts frontend/src/api/client.test.ts
git commit -m "feat(frontend): typed fetch client + query client (no auth header)"
```

---

### Task 6: Translator API calls + React Query hooks

**Files:**
- Create: `frontend/src/api/translator.ts`, `frontend/src/hooks/useLanguages.ts`,
  `frontend/src/hooks/useTranslate.ts`
- Test: `frontend/src/api/translator.test.ts`

**Interfaces:**
- Consumes: `apiGet`/`apiPost` (Task 5), the `types.ts` types.
- Produces: `getLanguages(signal?)`, `translate(req, signal?)` (translator.ts);
  `useLanguages()` (sorted by name, `staleTime: Infinity`), `useTranslate()`
  (mutation). Consumed by Task 9.

- [ ] **Step 1: Write the failing test — `frontend/src/api/translator.test.ts`**

```ts
import { afterEach, describe, expect, it, vi } from 'vitest'
import { getLanguages, translate } from './translator'

afterEach(() => vi.restoreAllMocks())

function mockJson(body: unknown) {
  vi.stubGlobal(
    'fetch',
    vi.fn(
      async () =>
        new Response(JSON.stringify(body), {
          status: 200,
          headers: { 'content-type': 'application/json' },
        }),
    ),
  )
}

describe('translator api', () => {
  it('getLanguages returns the decoded list', async () => {
    mockJson([
      { code: 'en', name: 'English' },
      { code: 'fr', name: 'French' },
    ])
    const langs = await getLanguages()
    expect(langs.map((l) => l.code)).toEqual(['en', 'fr'])
  })

  it('translate posts the request and returns the response', async () => {
    mockJson({ translation: 'Bonjour', detected_language: { name: 'English', flag: '🇬🇧' } })
    const res = await translate({ text: 'Hello', target_lang: 'fr' })
    expect(res.translation).toBe('Bonjour')
    expect(res.detected_language.name).toBe('English')
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd frontend && pnpm test -- translator`
Expected: FAIL — `Cannot find module './translator'`.

- [ ] **Step 3: Create `frontend/src/api/translator.ts`**

```ts
import { apiGet, apiPost } from './client'
import type { Language, TranslateRequest, TranslateResponse } from './types'

export function getLanguages(signal?: AbortSignal): Promise<Language[]> {
  return apiGet<Language[]>('/languages', signal)
}

export function translate(req: TranslateRequest, signal?: AbortSignal): Promise<TranslateResponse> {
  return apiPost<TranslateResponse>('/translate', req, signal)
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd frontend && pnpm test -- translator`
Expected: PASS (2 tests).

- [ ] **Step 5: Create `frontend/src/hooks/useLanguages.ts`**

```ts
import { useQuery } from '@tanstack/react-query'
import { getLanguages } from '../api/translator'
import type { Language } from '../api/types'

/** Load the supported languages once, sorted by display name. */
export function useLanguages() {
  return useQuery({
    queryKey: ['languages'],
    queryFn: ({ signal }) => getLanguages(signal),
    staleTime: Infinity,
    select: (langs: Language[]) => [...langs].sort((a, b) => a.name.localeCompare(b.name)),
  })
}
```

- [ ] **Step 6: Create `frontend/src/hooks/useTranslate.ts`**

```ts
import { useMutation } from '@tanstack/react-query'
import { translate } from '../api/translator'
import type { TranslateRequest, TranslateResponse } from '../api/types'

/** Mutation wrapper around POST /translate. */
export function useTranslate() {
  return useMutation<TranslateResponse, Error, TranslateRequest>({
    mutationFn: (req) => translate(req),
  })
}
```

- [ ] **Step 7: Type-check then commit**

Run: `cd frontend && pnpm typecheck`
Expected: PASS.

```bash
git add frontend/src/api/translator.ts frontend/src/api/translator.test.ts \
  frontend/src/hooks/useLanguages.ts frontend/src/hooks/useTranslate.ts
git commit -m "feat(frontend): translator api calls + languages/translate hooks"
```

---

### Task 7: Presentational components — selects, file input, detected-language banner

**Files:**
- Create: `frontend/src/components/LanguageSelect.tsx`,
  `frontend/src/components/FileTextInput.tsx`,
  `frontend/src/components/DetectedLanguageBanner.tsx`
- Test: `frontend/src/components/LanguageSelect.test.tsx`

**Interfaces:**
- Produces: `LanguageSelect` + exported `AUTO_DETECT` sentinel; `FileTextInput`
  (`onText`, `onError`); `DetectedLanguageBanner` (`name`, `flag`, `auto`).
  Consumed by Task 9.

- [ ] **Step 1: Write the failing test — `frontend/src/components/LanguageSelect.test.tsx`**

```tsx
import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AUTO_DETECT, LanguageSelect } from './LanguageSelect'

const LANGS = [
  { code: 'en', name: 'English' },
  { code: 'fr', name: 'French' },
]

describe('LanguageSelect', () => {
  it('renders an Auto-detect option when includeAuto is set', () => {
    render(
      <LanguageSelect id="source" label="Source language" value={AUTO_DETECT}
        onChange={() => {}} languages={LANGS} includeAuto />,
    )
    expect(screen.getByRole('option', { name: 'Auto-detect' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'French' })).toBeInTheDocument()
  })

  it('omits Auto-detect by default and reports selection by code', async () => {
    const onChange = vi.fn()
    render(
      <LanguageSelect id="target" label="Target language" value="en"
        onChange={onChange} languages={LANGS} />,
    )
    expect(screen.queryByRole('option', { name: 'Auto-detect' })).toBeNull()
    await userEvent.selectOptions(screen.getByLabelText('Target language'), 'fr')
    expect(onChange).toHaveBeenCalledWith('fr')
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd frontend && pnpm test -- LanguageSelect`
Expected: FAIL — `Cannot find module './LanguageSelect'`.

- [ ] **Step 3: Create `frontend/src/components/LanguageSelect.tsx`**

```tsx
import { Select } from '@infra/ui'
import type { Language } from '../api/types'

/** Sentinel value for the source selector's "Auto-detect" choice. */
export const AUTO_DETECT = '__auto__'

interface LanguageSelectProps {
  id: string
  label: string
  value: string
  onChange: (value: string) => void
  languages: Language[]
  includeAuto?: boolean
}

export function LanguageSelect({
  id,
  label,
  value,
  onChange,
  languages,
  includeAuto = false,
}: LanguageSelectProps) {
  return (
    <label htmlFor={id} className="flex flex-col gap-1 text-sm text-muted-foreground">
      {label}
      <Select id={id} value={value} onChange={(e) => onChange(e.target.value)}>
        {includeAuto && <option value={AUTO_DETECT}>Auto-detect</option>}
        {languages.map((lang) => (
          <option key={lang.code} value={lang.code}>
            {lang.name}
          </option>
        ))}
      </Select>
    </label>
  )
}
```

- [ ] **Step 4: Create `frontend/src/components/FileTextInput.tsx`**

```tsx
import type { ChangeEvent } from 'react'

interface FileTextInputProps {
  onText: (text: string) => void
  onError: (message: string) => void
}

/** A `.txt` picker that reads the file client-side and emits its UTF-8 text. */
export function FileTextInput({ onText, onError }: FileTextInputProps) {
  function handleChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => onText(typeof reader.result === 'string' ? reader.result : '')
    reader.onerror = () => onError('Could not read the file as UTF-8 text.')
    reader.readAsText(file, 'utf-8')
    e.target.value = '' // allow re-selecting the same file
  }
  return (
    <label className="inline-flex cursor-pointer items-center gap-2 text-sm text-muted-foreground hover:text-foreground">
      <span aria-hidden>📎</span>
      Upload .txt
      <input type="file" accept=".txt,text/plain" className="sr-only" onChange={handleChange} />
    </label>
  )
}
```

- [ ] **Step 5: Create `frontend/src/components/DetectedLanguageBanner.tsx`**

```tsx
import { Banner } from '@infra/ui'

interface DetectedLanguageBannerProps {
  name: string
  flag: string
  auto: boolean
}

/** Shows the detected (auto) or explicit source language with its flag. */
export function DetectedLanguageBanner({ name, flag, auto }: DetectedLanguageBannerProps) {
  const label = auto ? 'Detected source language' : 'Source language'
  return (
    <Banner variant="info">
      <span className="text-muted-foreground">{label}: </span>
      <span>
        {flag} {name}
      </span>
    </Banner>
  )
}
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `cd frontend && pnpm test -- LanguageSelect`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/LanguageSelect.tsx frontend/src/components/LanguageSelect.test.tsx \
  frontend/src/components/FileTextInput.tsx frontend/src/components/DetectedLanguageBanner.tsx
git commit -m "feat(frontend): language selects, .txt input, detected-language banner"
```

---

### Task 8: Output field with hover-reveal copy button

**Files:**
- Create: `frontend/src/components/OutputField.tsx`
- Test: `frontend/src/components/OutputField.test.tsx`

**Interfaces:**
- Produces: `OutputField` (`value`, `placeholder?`). Renders a read-only result
  region with a copy button revealed on hover/focus. Consumed by Task 9.

- [ ] **Step 1: Write the failing test — `frontend/src/components/OutputField.test.tsx`**

```tsx
import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { OutputField } from './OutputField'

afterEach(() => vi.restoreAllMocks())

describe('OutputField', () => {
  it('shows the placeholder and no copy button when empty', () => {
    render(<OutputField value="" placeholder="Translation will appear here." />)
    expect(screen.getByText('Translation will appear here.')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /copy/i })).toBeNull()
  })

  it('copies the value to the clipboard when the button is clicked', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal('navigator', { clipboard: { writeText } })
    render(<OutputField value="Bonjour le monde" />)
    await userEvent.click(screen.getByRole('button', { name: /copy translation/i }))
    expect(writeText).toHaveBeenCalledWith('Bonjour le monde')
    expect(screen.getByRole('button', { name: /copied/i })).toBeInTheDocument()
  })
})
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd frontend && pnpm test -- OutputField`
Expected: FAIL — `Cannot find module './OutputField'`.

- [ ] **Step 3: Create `frontend/src/components/OutputField.tsx`**

```tsx
import { useState } from 'react'
import { Button, cn } from '@infra/ui'

interface OutputFieldProps {
  value: string
  placeholder?: string
}

/** Read-only translation output with a copy button revealed on hover/focus. */
export function OutputField({ value, placeholder }: OutputFieldProps) {
  const [copied, setCopied] = useState(false)

  async function copy() {
    await navigator.clipboard.writeText(value)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1500)
  }

  return (
    <div className="group relative h-72 overflow-auto rounded-md border border-border bg-muted/30 p-3">
      {value ? (
        <p className="whitespace-pre-wrap text-sm text-foreground">{value}</p>
      ) : (
        <p className="text-sm text-muted-foreground">{placeholder}</p>
      )}
      {value && (
        <Button
          type="button"
          variant="secondary"
          size="sm"
          aria-label={copied ? 'Copied' : 'Copy translation'}
          onClick={copy}
          className={cn(
            'absolute right-2 top-2 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100',
            copied && 'opacity-100',
          )}
        >
          {copied ? '✓ Copied' : 'Copy'}
        </Button>
      )}
    </div>
  )
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd frontend && pnpm test -- OutputField`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/OutputField.tsx frontend/src/components/OutputField.test.tsx
git commit -m "feat(frontend): output field with hover-reveal copy button"
```

---

### Task 9: TranslatePanel + App wiring + full verification

**Files:**
- Create: `frontend/src/components/TranslatePanel.tsx`
- Test: `frontend/src/components/TranslatePanel.test.tsx`
- Modify: `frontend/src/App.tsx` (replace placeholder)

**Interfaces:**
- Consumes: `useLanguages`, `useTranslate` (Task 6); `LanguageSelect`/`AUTO_DETECT`,
  `FileTextInput`, `DetectedLanguageBanner` (Task 7); `OutputField` (Task 8);
  `Button`, `Banner`, `Spinner` from `@infra/ui`; `queryClient` (Task 5).

- [ ] **Step 1: Write the failing test — `frontend/src/components/TranslatePanel.test.tsx`**

```tsx
import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClientProvider } from '@tanstack/react-query'
import { QueryClient } from '@tanstack/react-query'
import { TranslatePanel } from './TranslatePanel'

afterEach(() => vi.restoreAllMocks())

function renderPanel() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <TranslatePanel />
    </QueryClientProvider>,
  )
}

function jsonResponse(body: unknown) {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  })
}

describe('TranslatePanel', () => {
  it('loads languages, translates, and shows output + detected language', async () => {
    const fetchFn = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/v1/languages')) {
        return jsonResponse([
          { code: 'en', name: 'English' },
          { code: 'fr', name: 'French' },
        ])
      }
      return jsonResponse({
        translation: 'Bonjour le monde',
        detected_language: { name: 'English', flag: '🇬🇧' },
      })
    })
    vi.stubGlobal('fetch', fetchFn)

    renderPanel()

    const input = await screen.findByLabelText('Text to translate')
    await userEvent.type(input, 'Hello world')
    await userEvent.click(screen.getByRole('button', { name: 'Translate' }))

    expect(await screen.findByText('Bonjour le monde')).toBeInTheDocument()
    await waitFor(() =>
      expect(screen.getByText(/Detected source language/)).toBeInTheDocument(),
    )
    // POST body omits source_lang while Auto-detect is selected.
    const postCall = fetchFn.mock.calls.find(([u]) => String(u).endsWith('/translate'))!
    const body = JSON.parse((postCall[1] as RequestInit).body as string)
    expect(body).toMatchObject({ text: 'Hello world', target_lang: 'en' })
    expect(body.source_lang).toBeUndefined()
  })
})
```

> The default target resolves to `English` here because `VITE_DEFAULT_TARGET_LANGUAGE` is unset in tests (fallback) and English sorts/loads in the list.

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd frontend && pnpm test -- TranslatePanel`
Expected: FAIL — `Cannot find module './TranslatePanel'`.

- [ ] **Step 3: Create `frontend/src/components/TranslatePanel.tsx`**

```tsx
import { useEffect, useState } from 'react'
import { Banner, Button, Spinner } from '@infra/ui'
import { useLanguages } from '../hooks/useLanguages'
import { useTranslate } from '../hooks/useTranslate'
import { AUTO_DETECT, LanguageSelect } from './LanguageSelect'
import { FileTextInput } from './FileTextInput'
import { OutputField } from './OutputField'
import { DetectedLanguageBanner } from './DetectedLanguageBanner'

const DEFAULT_TARGET = import.meta.env.VITE_DEFAULT_TARGET_LANGUAGE ?? 'English'

export function TranslatePanel() {
  const languagesQuery = useLanguages()
  const translation = useTranslate()

  const [source, setSource] = useState(AUTO_DETECT)
  const [target, setTarget] = useState('')
  const [text, setText] = useState('')
  const [fileError, setFileError] = useState<string | null>(null)

  const languages = languagesQuery.data ?? []

  // Initialize the target default once languages load.
  useEffect(() => {
    if (target === '' && languages.length > 0) {
      const def = languages.find((l) => l.name === DEFAULT_TARGET) ?? languages[0]
      setTarget(def.code)
    }
  }, [languages, target])

  function onTranslate() {
    setFileError(null)
    translation.mutate({
      text: text.trim(),
      target_lang: target,
      source_lang: source === AUTO_DETECT ? undefined : source,
    })
  }

  if (languagesQuery.isLoading) return <Spinner label="Loading languages…" />
  if (languagesQuery.error) {
    return <Banner variant="danger">Could not load languages: {String(languagesQuery.error)}</Banner>
  }

  const result = translation.data
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <LanguageSelect
          id="source"
          label="Source language"
          value={source}
          onChange={setSource}
          languages={languages}
          includeAuto
        />
        <LanguageSelect
          id="target"
          label="Target language"
          value={target}
          onChange={setTarget}
          languages={languages}
        />
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2">
          <textarea
            aria-label="Text to translate"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Enter text to translate…"
            className="h-72 w-full resize-none rounded-md border border-border bg-background p-3 text-sm text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          />
          <FileTextInput onText={setText} onError={setFileError} />
        </div>
        <OutputField value={result?.translation ?? ''} placeholder="Translation will appear here." />
      </div>

      <div className="flex items-center gap-3">
        <Button
          type="button"
          onClick={onTranslate}
          disabled={text.trim().length === 0 || target === '' || translation.isPending}
        >
          Translate
        </Button>
        {translation.isPending && <Spinner label="Translating…" />}
      </div>

      {fileError && <Banner variant="danger">{fileError}</Banner>}
      {translation.error && (
        <Banner variant="danger">Translation failed: {String(translation.error)}</Banner>
      )}
      {result && (
        <DetectedLanguageBanner
          name={result.detected_language.name}
          flag={result.detected_language.flag}
          auto={source === AUTO_DETECT}
        />
      )}
    </div>
  )
}
```

- [ ] **Step 4: Replace `frontend/src/App.tsx`**

```tsx
import { QueryClientProvider } from '@tanstack/react-query'
import { queryClient } from './api/queryClient'
import { TranslatePanel } from './components/TranslatePanel'

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-full">
        <header className="border-b border-border px-6 py-4">
          <h1 className="text-lg font-semibold">🌐 Translator</h1>
        </header>
        <main className="mx-auto max-w-5xl px-6 py-8">
          <TranslatePanel />
        </main>
      </div>
    </QueryClientProvider>
  )
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd frontend && pnpm test -- TranslatePanel`
Expected: PASS.

- [ ] **Step 6: Full frontend gate — lint, typecheck, all tests, build**

Run: `cd frontend && pnpm lint && pnpm typecheck && pnpm test && pnpm build`
Expected: all green; `frontend/dist/` written.

- [ ] **Step 7: Manual end-to-end check against the running backend**

Run (terminal A): `OPENAI_API_BASE=<your-endpoint>/v1 uv run uvicorn translator.main:app --reload`
Run (terminal B): `cd frontend && pnpm dev`
Open `http://localhost:5173`. Expected: language dropdowns populate (proxy forwarded `/api/v1/languages`), a translation round-trips and renders, the detected-language banner appears, and the copy button reveals on hover and copies.

- [ ] **Step 8: Commit**

```bash
git add frontend/src/components/TranslatePanel.tsx \
  frontend/src/components/TranslatePanel.test.tsx frontend/src/App.tsx
git commit -m "feat(frontend): translate panel wired end-to-end"
```

---

### Task 10: nginx config (same-origin API proxy)

**Files:**
- Create: `frontend/nginx/security-headers.conf`, `frontend/nginx/default.conf`

> translator has no SSE and no file upload over the wire (the `.txt` is read in the browser), so the proxy is a single `/api/` location — simpler than the sibling configs.

- [ ] **Step 1: Create `frontend/nginx/security-headers.conf`**

```nginx
add_header X-Content-Type-Options "nosniff" always;
add_header X-Frame-Options "DENY" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Permissions-Policy "camera=(), microphone=(), geolocation=()" always;
add_header Content-Security-Policy "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; connect-src 'self'; font-src 'self' data:; object-src 'none'; base-uri 'self'; frame-ancestors 'none'" always;
```

- [ ] **Step 2: Create `frontend/nginx/default.conf`**

```nginx
server {
    listen 80;
    server_name _;
    resolver 127.0.0.11 valid=30s ipv6=off;
    root /usr/share/nginx/html;

    set $backend_upstream "backend:8000";

    # All API calls proxy same-origin to the FastAPI backend.
    location /api/ {
        include /etc/nginx/security-headers.conf;
        proxy_pass http://$backend_upstream$request_uri;
        proxy_set_header Host $host;
    }

    # Hashed, immutable build assets.
    location /assets/ {
        include /etc/nginx/security-headers.conf;
        add_header Cache-Control "public, max-age=31536000, immutable" always;
        try_files $uri =404;
    }

    # SPA fallback for client-side routing.
    location / {
        include /etc/nginx/security-headers.conf;
        add_header Cache-Control "no-cache, must-revalidate" always;
        try_files $uri /index.html;
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add frontend/nginx/security-headers.conf frontend/nginx/default.conf
git commit -m "feat(frontend): nginx same-origin /api proxy + SPA fallback"
```

---

### Task 11: Dockerfile (node build → nginx serve)

**Files:**
- Modify: `docker/Dockerfile.frontend` (replace the Streamlit image entirely)

- [ ] **Step 1: Replace the full contents of `docker/Dockerfile.frontend`**

```dockerfile
# syntax=docker/dockerfile:1
ARG NODE_IMAGE=docker.io/library/node:20-alpine
ARG NGINX_IMAGE=docker.io/library/nginx:1.27-alpine

FROM ${NODE_IMAGE} AS builder
WORKDIR /build
RUN corepack enable && corepack prepare pnpm@9.12.0 --activate
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/ ./
ARG VITE_DEFAULT_TARGET_LANGUAGE=English
ENV VITE_DEFAULT_TARGET_LANGUAGE=${VITE_DEFAULT_TARGET_LANGUAGE}
RUN pnpm build

FROM ${NGINX_IMAGE}
RUN rm /etc/nginx/conf.d/default.conf
COPY frontend/nginx/security-headers.conf /etc/nginx/security-headers.conf
COPY frontend/nginx/default.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /build/dist /usr/share/nginx/html
EXPOSE 80
```

- [ ] **Step 2: Build the image to verify the Dockerfile**

Run: `DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.frontend -t translator-frontend:dev .`
Expected: build succeeds; final image is nginx-based serving the built SPA.

- [ ] **Step 3: Commit**

```bash
git add docker/Dockerfile.frontend
git commit -m "feat(docker): node→nginx frontend image replacing Streamlit"
```

---

### Task 12: Compose wiring (swap the frontend service)

**Files:**
- Modify: `docker/compose.yaml:55-73` (frontend service)
- Modify: `docker/compose.override.yaml:14-16` (frontend dev port)

- [ ] **Step 1: Replace the `frontend` service in `docker/compose.yaml`** (lines 55–73) with:

```yaml
  frontend:
    image: translator-frontend:${TRANSLATOR_VERSION:-latest}
    build:
      context: ..
      dockerfile: docker/Dockerfile.frontend
      args:
        VITE_DEFAULT_TARGET_LANGUAGE: ${DEFAULT_TARGET_LANGUAGE:-English}
    logging: *default-logging
    environment:
      <<: *no-proxy-env
    env_file: *env-file
    networks:
      translator-net:
    depends_on:
      - backend
    expose:
      - "80"
    restart: unless-stopped
```

- [ ] **Step 2: Update the frontend dev port in `docker/compose.override.yaml`** (lines 14–16) from `"${TRANSLATOR_HOST_PORT:-8501}:8501"` to:

```yaml
  frontend:
    ports:
      - "${TRANSLATOR_HOST_PORT:-8501}:80"
```

- [ ] **Step 3: Validate compose interpolation**

Run: `docker compose -f docker/compose.yaml -f docker/compose.override.yaml config | sed -n '/frontend:/,/restart/p'`
Expected: `frontend` resolves with the build arg, `expose: 80`, and the `${TRANSLATOR_HOST_PORT:-8501}:80` mapping; no `BACKEND_URL`/`STREAMLIT_*`/`DEFAULT_TARGET_LANGUAGE` env remain.

- [ ] **Step 4: Commit**

```bash
git add docker/compose.yaml docker/compose.override.yaml
git commit -m "feat(docker): point compose frontend at the React/nginx image"
```

---

### Task 13: Makefile help text + dockerized smoke test

**Files:**
- Modify: `Makefile:23` (help text)

- [ ] **Step 1: Update the help line in `Makefile`** — change line 23 from:

```makefile
	@echo "translator — FastAPI backend + Streamlit frontend."
```

to:

```makefile
	@echo "translator — FastAPI backend + React SPA (nginx)."
```

- [ ] **Step 2: Build both images via the make workflow**

Run: `make build`
Expected: builds `translator-backend` and `translator-frontend` (React/nginx) without error.

- [ ] **Step 3: Smoke test the dev stack**

Run: `make network && OPENAI_API_BASE=<your-endpoint>/v1 make up-dev`
Open `http://localhost:8501`. Expected: the SPA loads, language dropdowns populate, and a translation round-trips through nginx → `backend:8000` (proving the same-origin proxy). Then `make down`.

- [ ] **Step 4: Commit**

```bash
git add Makefile
git commit -m "docs(make): describe the React SPA frontend"
```

---

## Definition of done (Plan 01)

- `cd frontend && pnpm lint && pnpm typecheck && pnpm test && pnpm build` all pass.
- `OPENAI_API_BASE=... uv run pytest` passes (routes under `/api/v1`, `/health` present, CORS gone).
- `make build` builds backend + React/nginx frontend; `make up-dev` serves the SPA on `:${TRANSLATOR_HOST_PORT:-8501}`, dropdowns populate, a translation round-trips, the detected-language banner shows, and the hover copy button works.
- Streamlit `app.py` still exists on disk (removed in Plan 02) but is no longer wired into Docker.

## Self-review (against the design spec)

- **Spec coverage:** `/api/v1` + `/health` + CORS removal (T1); Vite/React/TS/pnpm scaffold (T2–T3); Tailwind v4 + `@infra/ui` theme + `cn` (T4); typed client with **no auth header** (T5); languages/translate hooks (T6); parity components — selects, `.txt` client-side read, detected banner (T7); hover copy button (T8); full parity panel + E2E (T9); same-origin nginx (T10); node→nginx Dockerfile (T11); compose swap + build-arg default language (T12); Makefile + dockerized smoke (T13). Teardown of Streamlit/deps and docs are Plan 02 by design.
- **Placeholder scan:** none — every code/config step is complete; `<your-endpoint>` in manual steps is an operator-supplied runtime value, not plan content.
- **Type consistency:** `apiGet`/`apiPost`/`ApiError` (T5) are consumed unchanged by `translator.ts` (T6); `Language`/`TranslateRequest`/`TranslateResponse` (T5) flow through hooks (T6) into `TranslatePanel` (T9); `AUTO_DETECT` is defined once (T7) and imported by `TranslatePanel` (T9); `@infra/ui` imports (`Button`, `Banner`, `Spinner`, `Select`, `cn`) match the package's real exports.

## Carry-forward to Plan 02

- Delete `translator/app.py`; remove the `[dependency-groups].frontend` group and drop `streamlit`, `requests`, `types-requests` from base `dependencies`; `uv lock` + verify `uv run pytest`.
- Rewrite README architecture table / dev commands / ports; update CLAUDE.md (architecture, commands, docker shape, env table, drop the stale "frontend/ Node app no longer exists" note — it exists again); update `.env.example` (`DEFAULT_TARGET_LANGUAGE` is now a build arg; `TRANSLATOR_HOST_PORT` maps to `:80`).
- Flag the parent `infra/CLAUDE.md` table (`translator → FastAPI + Streamlit`) as an out-of-repo follow-up.
