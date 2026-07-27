import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClientProvider, QueryClient } from '@tanstack/react-query'
import { TranslatePanel } from './TranslatePanel'
import { LanguageContext } from '../i18n/LanguageContext'

afterEach(() => vi.restoreAllMocks())

function renderPanel() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <TranslatePanel />
    </QueryClientProvider>,
  )
}

function renderPanelInLanguage(lang: 'en' | 'de') {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={qc}>
      <LanguageContext.Provider value={lang}>
        <TranslatePanel />
      </LanguageContext.Provider>
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
    const fetchFn = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
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

  it('sends the user-selected target language when overridden', async () => {
    const fetchFn = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
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

    // Wait for languages to load before selecting the override.
    const input = await screen.findByLabelText('Text to translate')
    await userEvent.selectOptions(screen.getByLabelText('Target language'), 'fr')
    await userEvent.type(input, 'Hello world')
    await userEvent.click(screen.getByRole('button', { name: 'Translate' }))

    expect(await screen.findByText('Bonjour le monde')).toBeInTheDocument()
    const postCall = fetchFn.mock.calls.find(([u]) => String(u).endsWith('/translate'))!
    const body = JSON.parse((postCall[1] as RequestInit).body as string)
    expect(body).toMatchObject({ text: 'Hello world', target_lang: 'fr' })
  })

  it('renders a generic error banner on translate failure, never the raw detail', async () => {
    const fetchFn = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/v1/languages')) {
        return jsonResponse([
          { code: 'en', name: 'English' },
          { code: 'fr', name: 'French' },
        ])
      }
      return new Response(JSON.stringify({ detail: 'boom' }), {
        status: 500,
        headers: { 'content-type': 'application/json' },
      })
    })
    vi.stubGlobal('fetch', fetchFn)

    renderPanel()

    const input = await screen.findByLabelText('Text to translate')
    await userEvent.type(input, 'Hello world')
    await userEvent.click(screen.getByRole('button', { name: 'Translate' }))

    expect(
      await screen.findByText('Something went wrong (500). Please try again or contact support.'),
    ).toBeInTheDocument()
    expect(screen.queryByText(/boom/)).not.toBeInTheDocument()
  })

  it('renders German chrome when language is de', async () => {
    const fetchFn = vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input)
      if (url.endsWith('/api/v1/languages')) {
        return jsonResponse([
          { code: 'en', name: 'English' },
          { code: 'fr', name: 'French' },
        ])
      }
      return jsonResponse({})
    })
    vi.stubGlobal('fetch', fetchFn)

    renderPanelInLanguage('de')

    await screen.findByLabelText('Zu übersetzender Text')
    expect(screen.getByRole('button', { name: 'Übersetzen' })).toBeInTheDocument()
  })
})
