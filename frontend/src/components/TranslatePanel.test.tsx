import { afterEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClientProvider, QueryClient } from '@tanstack/react-query'
import { clearAllPurposes } from '@infra/ui'
import { AppPurposeBar } from './AppPurposeBar'
import { TranslatePanel } from './TranslatePanel'
import { LanguageContext } from '../i18n/LanguageContext'

afterEach(() => {
  vi.restoreAllMocks()
  clearAllPurposes()
  sessionStorage.clear()
})

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
      <LanguageContext value={lang}>
        <TranslatePanel />
      </LanguageContext>
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
    await userEvent.click(screen.getByRole('combobox', { name: 'Target language' }))
    await userEvent.click(screen.getByRole('option', { name: 'French' }))
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

  it.each([
    {
      lang: 'en',
      textLabel: 'Text to translate',
      translate: 'Translate',
      cancel: 'Cancel',
      message: 'This action needs a stated purpose. Please state one in the header and try again.',
    },
    {
      lang: 'de',
      textLabel: 'Zu übersetzender Text',
      translate: 'Übersetzen',
      cancel: 'Abbrechen',
      message:
        'Für diese Aktion muss ein Zweck angegeben werden – Bitte geben Sie ihn oben in der Kopfzeile an und versuchen Sie es erneut.',
    },
  ] as const)(
    'a declined purpose prompt says what is missing, not the generic error ($lang)',
    async ({ lang, textLabel, translate, cancel, message }) => {
      const fetchFn = vi.fn(async (input: RequestInfo | URL) => {
        const url = String(input)
        if (url.endsWith('/api/v1/languages')) {
          return jsonResponse([
            { code: 'en', name: 'English' },
            { code: 'fr', name: 'French' },
          ])
        }
        return new Response(JSON.stringify({ detail: 'This action requires a stated purpose.' }), {
          status: 428,
          headers: { 'content-type': 'application/json' },
        })
      })
      vi.stubGlobal('fetch', fetchFn)
      const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })
      render(
        <QueryClientProvider client={qc}>
          <LanguageContext value={lang}>
            <AppPurposeBar required />
            <TranslatePanel />
          </LanguageContext>
        </QueryClientProvider>,
      )

      await userEvent.type(await screen.findByLabelText(textLabel), 'Hello Alice Example')
      await userEvent.click(screen.getByRole('button', { name: translate }))
      const dialog = await screen.findByRole('dialog')
      await userEvent.click(within(dialog).getByRole('button', { name: cancel }))

      expect(await screen.findByText(message)).toBeInTheDocument()
      expect(screen.queryByText(/428/)).not.toBeInTheDocument()
      const translateCalls = fetchFn.mock.calls.filter(([u]) => String(u).endsWith('/translate'))
      expect(translateCalls).toHaveLength(1)
    },
  )

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

function stubLanguages(translate: () => Promise<Response> = () => new Promise(() => {})) {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL) => {
      if (String(input).endsWith('/api/v1/languages')) {
        return jsonResponse([
          { code: 'en', name: 'English' },
          { code: 'fr', name: 'French' },
        ])
      }
      return translate()
    }),
  )
}

describe('TranslatePanel actions', () => {
  it('upload and translate share one full-width bar, translate last', async () => {
    stubLanguages()
    renderPanel()
    const translate = await screen.findByRole('button', { name: 'Translate' })
    const bar = translate.parentElement!
    expect(bar).toHaveClass('flex-wrap', '*:flex-[1_1_8rem]')
    expect(bar).toContainElement(screen.getByRole('button', { name: 'Upload .txt' }))
    expect(bar.lastElementChild).toBe(translate)
  })

  it('shows the pending state on the button itself', async () => {
    stubLanguages()
    renderPanel()
    await userEvent.type(await screen.findByLabelText('Text to translate'), 'Hello')
    await userEvent.click(screen.getByRole('button', { name: 'Translate' }))
    expect(await screen.findByRole('button', { name: 'Translating…' })).toBeDisabled()
    expect(screen.queryByRole('status')).toBeNull()
  })

  it('puts a file error above the bar', async () => {
    stubLanguages()
    class FailingReader {
      onload: (() => void) | null = null
      onerror: (() => void) | null = null
      result: string | null = null
      readAsText() {
        setTimeout(() => this.onerror?.(), 0)
      }
    }
    vi.stubGlobal('FileReader', FailingReader)
    const { container } = renderPanel()
    const translate = await screen.findByRole('button', { name: 'Translate' })
    const input = container.querySelector('input[type="file"]') as HTMLInputElement
    fireEvent.change(input, { target: { files: [new File(['x'], 'kaputt.txt')] } })
    const banner = await screen.findByText(/could not read the file/i)
    expect(banner.compareDocumentPosition(translate) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
  })
})
