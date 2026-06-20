import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClientProvider, QueryClient } from '@tanstack/react-query'
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
})
