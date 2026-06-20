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
