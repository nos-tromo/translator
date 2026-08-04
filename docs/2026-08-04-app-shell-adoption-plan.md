# translator — AppShell Adoption Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adopt `@infra/ui` v0.9.0 in the translator SPA — `AppShell` (fixed
chrome, no sidebar), `PageHeader`, sign-out via `UserMenu` — and rework the
translate view into full-height equal soft-fill panes with banners next to
their controls.

**Architecture:** Plan 2 of the federation rollout defined in the infra-ui
repo (`docs/2026-08-04-app-shell-federation-design.md` there). Frontend-only:
`App.tsx` swaps its hand-rolled header+main for `AppShell`+`PageHeader`;
`TranslatePanel`/`OutputField` become a flex column that fills the canvas
height. No backend changes, no route changes, no new features beyond
sign-out.

**Tech Stack:** React 19 + Vite + Tailwind v4 + `@infra/ui` (tag-pinned pnpm
git dep) + vitest.

## Global Constraints

- All frontend commands run inside `frontend/` with pnpm.
- Functionality preserved: same endpoints, language auto-detect, file
  upload, copy button, i18n (en/de), theme toggle, user display; sign-out is
  the ONE addition (via AppShell's built-in `UserMenu`, default
  `/auth/logout`).
- Semantic tokens only; the tile language for panes: `rounded-lg border
  border-border bg-muted` fill, no shadows.
- i18n: every new key exists in BOTH `src/i18n/en.ts` and `src/i18n/de.ts`
  (`i18n.test.ts` enforces parity). German uses Sie-form conventions.
- Known accepted limitation: `AppShell` v0.9.0 does not forward a
  `menuLabel` to `UserMenu`, so the menu button's aria-label prefix stays
  the English default "Account" in both locales. Do not work around it.
- Tests stay behavior-based (roles/text, not classes) except where a class
  IS the behavior under test.
- Confidentiality: synthetic data only; no local machine paths in anything
  committed.
- Working branch: `feature/app-shell` off current `main`.

---

### Task 1: Bump the `@infra/ui` pin to v0.9.0

**Files:**
- Modify: `frontend/package.json:18`
- Modify: `frontend/pnpm-lock.yaml` (via install)

**Interfaces:**
- Produces: `@infra/ui` v0.9.0 in `node_modules` — exports `AppShell`,
  `SidebarGroup`, `PageHeader`, `UserMenu`, `ThemeToggleLabels`, tile-style
  `Card`; `Shell` no longer exists; `AppHeader` still exported (still used
  until Task 3).

- [ ] **Step 1: Branch**

```bash
git checkout main && git pull && git checkout -b feature/app-shell
```

- [ ] **Step 2: Bump the pin** — in `frontend/package.json` change

```json
"@infra/ui": "github:nos-tromo/infra-ui#v0.8.1",
```

to

```json
"@infra/ui": "github:nos-tromo/infra-ui#v0.9.0",
```

- [ ] **Step 3: Install and run the existing gates**

```bash
cd frontend && pnpm install && pnpm lint && pnpm typecheck && pnpm test && pnpm build
```

Expected: all green — v0.9.0 keeps `AppHeader` exported, and translator
never imported `Shell` or `Card`, so nothing breaks at this pin bump.

- [ ] **Step 4: Commit**

```bash
git add frontend/package.json frontend/pnpm-lock.yaml
git commit -m "chore(frontend): bump @infra/ui to v0.9.0"
```

---

### Task 2: i18n keys for the shell

**Files:**
- Modify: `frontend/src/i18n/en.ts` (appheader + new page section)
- Modify: `frontend/src/i18n/de.ts` (same keys)

**Interfaces:**
- Produces the keys Task 3 consumes: `appheader.sign_out`, `page.title`,
  `page.caption` (translated via the existing `useT()` hook).

- [ ] **Step 1: Add to `frontend/src/i18n/en.ts`** — extend the appheader
  block and add a page block after it:

```ts
  'appheader.sign_out': 'Sign out',

  // page
  'page.title': 'Translate',
  'page.caption': 'Text translation',
```

- [ ] **Step 2: Add to `frontend/src/i18n/de.ts`** in the same positions:

```ts
  'appheader.sign_out': 'Abmelden',

  // page
  'page.title': 'Übersetzen',
  'page.caption': 'Textübersetzung',
```

- [ ] **Step 3: Run the parity test**

Run: `cd frontend && pnpm test src/i18n/i18n.test.ts`
Expected: PASS (both locales updated together).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/i18n/en.ts frontend/src/i18n/de.ts
git commit -m "feat(frontend): i18n keys for AppShell page header and sign-out"
```

---

### Task 3: Swap the shell — `AppShell` + `PageHeader` in `App.tsx`

**Files:**
- Modify: `frontend/src/App.tsx`
- Modify: `frontend/src/App.test.tsx`

**Interfaces:**
- Consumes: `AppShell` props `{ title, version?, user?, homeLabel?,
  themeLabels?, signOutLabel?, children }` and `PageHeader { title,
  caption? }` from `@infra/ui` v0.9.0; i18n keys from Task 2.
- Produces: the canvas content wrapper `<div className="flex h-full min-h-0
  flex-col p-8">` that Task 4's full-height `TranslatePanel` relies on.

- [ ] **Step 1: Update the tests first** — replace the four test bodies in
  `frontend/src/App.test.tsx` (mocks and setup stay unchanged). The old
  assertions used `AppHeader`'s testids; `AppShell` has none, and the user
  now renders as the `UserMenu` button:

```tsx
describe('App', () => {
  it('renders exactly one header row, with the version shown in it', async () => {
    getWhoami.mockResolvedValue({ username: null, display_name: null })
    render(<App />)
    expect(await screen.findByText('v1.2.3')).toBeInTheDocument()
    expect(screen.getAllByRole('banner')).toHaveLength(1)
    expect(await screen.findByTestId('translate-panel')).toBeInTheDocument()
  })

  it('shows the user menu with the display name when whoami resolves one', async () => {
    getWhoami.mockResolvedValue({ username: 'alex', display_name: 'Alex Example' })
    render(<App />)
    expect(await screen.findByRole('button', { name: /Alex Example/ })).toBeInTheDocument()
  })

  it('falls back to username in the user menu when whoami has no display name', async () => {
    getWhoami.mockResolvedValue({ username: 'alex', display_name: null })
    render(<App />)
    expect(await screen.findByRole('button', { name: /alex/ })).toBeInTheDocument()
  })

  it('omits the user menu when whoami has neither field', async () => {
    getWhoami.mockResolvedValue({ username: null, display_name: null })
    render(<App />)
    await screen.findByText('v1.2.3')
    expect(screen.queryByRole('button', { name: /account/i })).toBeNull()
  })
})
```

- [ ] **Step 2: Run to verify the new assertions fail against the old shell**

Run: `cd frontend && pnpm test src/App.test.tsx`
Expected: FAIL — old `AppHeader` renders the user as plain text, not a
button (tests 2-3); `findByText('v1.2.3')` may pass (version text exists),
that's fine.

- [ ] **Step 3: Rewrite `AppContent` in `frontend/src/App.tsx`** — imports
  change from `AppHeader` to `AppShell, PageHeader`; the returned JSX
  becomes:

```tsx
  return (
    <AppShell
      title="translator"
      version={data?.version ? `v${data.version}` : undefined}
      user={whoami?.display_name ?? whoami?.username ?? undefined}
      homeLabel={t('appheader.home')}
      themeLabels={{
        system: t('appheader.theme_system'),
        light: t('appheader.theme_light'),
        dark: t('appheader.theme_dark'),
      }}
      signOutLabel={t('appheader.sign_out')}
    >
      <div className="flex h-full min-h-0 flex-col p-8">
        <PageHeader title={t('page.title')} caption={t('page.caption')} />
        <TranslatePanel />
      </div>
    </AppShell>
  )
```

The old `<div className="min-h-full">`, `<AppHeader …/>`, and
`<main className="mx-auto max-w-5xl px-6 py-8">` wrappers are gone. Query
hooks and providers stay exactly as they are.

- [ ] **Step 4: Run the tests**

Run: `cd frontend && pnpm test src/App.test.tsx`
Expected: PASS (4/4).

- [ ] **Step 5: Full gates, commit**

Run: `cd frontend && pnpm lint && pnpm typecheck && pnpm test && pnpm build`
Expected: all green.

```bash
git add frontend/src/App.tsx frontend/src/App.test.tsx
git commit -m "feat(frontend): adopt AppShell + PageHeader (fixed chrome, sign-out menu)"
```

---

### Task 4: Full-height equal panes + banner placement

**Files:**
- Modify: `frontend/src/components/TranslatePanel.tsx:52-107` (returned JSX)
- Modify: `frontend/src/components/OutputField.tsx:13`
- Test: `frontend/src/components/TranslatePanel.test.tsx`,
  `frontend/src/components/OutputField.test.tsx` (existing suites are the
  spec — they must pass; only rewrite an assertion if it tests position or
  a class that legitimately changed, and say so in the report)

**Interfaces:**
- Consumes: the `flex h-full min-h-0 flex-col` canvas wrapper from Task 3.
- Produces: no API changes — `TranslatePanel` and `OutputField` keep their
  exact props and behavior.

- [ ] **Step 1: Rework the `TranslatePanel` returned JSX** (state, hooks,
  handlers, loading/error early-returns all stay untouched). Replace the
  `return (...)` block with:

```tsx
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4">
      <div className="grid gap-4 md:grid-cols-2">
        <LanguageSelect
          id="source"
          label={t('panel.source_language')}
          value={source}
          onChange={setSource}
          languages={languages}
          includeAuto
        />
        <LanguageSelect
          id="target"
          label={t('panel.target_language')}
          value={target}
          onChange={setTargetOverride}
          languages={languages}
        />
      </div>

      <div className="grid min-h-0 flex-1 gap-4 md:grid-cols-2">
        <div className="flex min-h-72 flex-col gap-2">
          <textarea
            aria-label={t('panel.text_aria')}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={t('panel.text_placeholder')}
            className="min-h-0 w-full flex-1 resize-none rounded-lg border border-border bg-muted p-3 text-sm text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          />
          <FileTextInput onText={setText} onError={setFileError} />
          {fileError && <Banner variant="danger">{fileError}</Banner>}
        </div>
        <OutputField value={result?.translation ?? ''} placeholder={t('panel.output_placeholder')} />
      </div>

      <div className="flex items-center gap-3">
        <Button
          type="button"
          onClick={onTranslate}
          disabled={text.trim().length === 0 || target === '' || translation.isPending}
        >
          {t('panel.translate')}
        </Button>
        {translation.isPending && <Spinner label={t('panel.translating')} />}
      </div>

      {translateError && <Banner variant="danger">{t(translateError.key, translateError.vars)}</Banner>}
      {result && (
        <DetectedLanguageBanner
          name={result.detected_language.name}
          flag={result.detected_language.flag}
          auto={source === AUTO_DETECT}
        />
      )}
    </div>
  )
```

What changed and why: root is a fill-height flex column (was `space-y-4`
with fixed `h-72` panes — the dead space below the fold); the pane grid is
`flex-1` so both panes grow to the canvas; the input becomes a soft-fill
tile (`rounded-lg bg-muted`) equal to the output (they were subtly
mismatched); `fileError` moved next to the file input that produces it;
`translateError`/detected-language sit directly under the Translate button.

- [ ] **Step 2: Match the `OutputField` surface** — in
  `frontend/src/components/OutputField.tsx` change the wrapper div class
  from

```
group relative h-72 overflow-auto rounded-md border border-border bg-muted/30 p-3
```

to

```
group relative h-full min-h-72 overflow-auto rounded-lg border border-border bg-muted p-3
```

- [ ] **Step 3: Run the component suites**

Run: `cd frontend && pnpm test src/components/TranslatePanel.test.tsx src/components/OutputField.test.tsx`
Expected: PASS — the suites assert behavior (labels, roles, copy flow,
error banners), none of which changed. If an assertion fails, fix ONLY
assertions that encode the old layout (e.g. a hardcoded class); never
weaken a behavioral assertion.

- [ ] **Step 4: Full gates, commit**

Run: `cd frontend && pnpm lint && pnpm typecheck && pnpm test && pnpm build`
Expected: all green.

```bash
git add frontend/src/components/TranslatePanel.tsx frontend/src/components/OutputField.tsx frontend/src/components/TranslatePanel.test.tsx frontend/src/components/OutputField.test.tsx
git commit -m "feat(frontend): full-height equal translate panes, banners beside their controls"
```

---

### Task 5: Accent unification, release bump, verify

**Files:**
- Modify: `frontend/src/styles/globals.css:5-24`
- Modify: `pyproject.toml:3` (`version = "1.2.0"` → `"1.3.0"`)

**Interfaces:**
- Consumes: nothing new. Produces: the release PR content (merge to `main`
  auto-mints `v1.3.0` via the shared release-tag workflow).

- [ ] **Step 1: Rewrite the accent blocks in
  `frontend/src/styles/globals.css`** — replace lines 5-24 (the `:root`,
  `:root:not([data-theme='dark'])`, and `@media` blocks) with the
  federation-standard shape (light default + explicit dark override —
  values unchanged, selector strategy unified with docint):

```css
:root {
  /* translator's single design-system knob; primary derives from this.
     Light default is darkened for WCAG AA as text on white (~4.5:1). */
  --app-accent: hsl(217 91% 45%);
}

:root[data-theme='dark'] {
  --app-accent: hsl(217 91% 60%);
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme]) {
    --app-accent: hsl(217 91% 60%);
  }
}
```

The `html, body, #root { height: 100% }` and `body` blocks below stay
unchanged (AppShell's `h-screen` relies on them).

- [ ] **Step 2: Bump the declared version** — `pyproject.toml`
  `[project].version` → `1.3.0` (this is the whole release action; the tag
  is auto-minted on merge).

- [ ] **Step 3: The full pre-push gate**

```bash
make verify
```

Expected: green (pre-commit ruff+pyrefly on the untouched backend, plus
`pnpm lint` + `pnpm build` for the frontend). Also run the frontend suite
once more: `cd frontend && pnpm test` — all green.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/styles/globals.css pyproject.toml
git commit -m "chore: unify accent declaration with the federation pattern; v1.3.0"
```

- [ ] **Step 5: STOP — do not push.** The controller runs the visual pass
  (dev bring-up, dark+light, wide+narrow) with the human before the PR is
  opened.
