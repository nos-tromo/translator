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
    expect(ri.headers).not.toHaveProperty('X-Auth-User')
  })

  it('throws ApiError carrying status and detail on non-2xx', async () => {
    mockFetch(422, { detail: 'invalid' })
    await expect(apiPost('/translate', { text: '' })).rejects.toMatchObject({
      name: 'ApiError',
      status: 422,
      detail: 'invalid',
    } satisfies Partial<ApiError>)
  })
})
