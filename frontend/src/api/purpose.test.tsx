import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { PURPOSE_HEADER, PurposeBar, clearAllPurposes, setPurpose } from '@infra/ui'
import { ApiError, PURPOSE_SCOPE, apiGet, apiPost } from './client'

const BODY = { text: 'Hello Alice Example', target_lang: 'fr' }

afterEach(() => {
  clearAllPurposes()
  sessionStorage.clear()
  vi.restoreAllMocks()
})

/** Stub fetch to answer with these statuses in turn, the last one from then on. */
function respondWith(...statuses: number[]) {
  const queue = [...statuses]
  const fn = vi.fn(async (_input: RequestInfo | URL, _init?: RequestInit) => {
    const status = queue.length > 1 ? queue.shift()! : queue[0]
    const body = status === 428 ? { detail: 'This action requires a stated purpose.' } : { ok: true }
    return new Response(JSON.stringify(body), {
      status,
      headers: { 'content-type': 'application/json' },
    })
  })
  vi.stubGlobal('fetch', fn)
  return fn
}

function sent(fn: ReturnType<typeof respondWith>, call: number) {
  const init = fn.mock.calls[call][1] as RequestInit
  return { headers: init.headers as Record<string, string>, body: init.body }
}

describe('every API request carries the stated purpose', () => {
  it('sends no X-Audit-Purpose while none is stated', async () => {
    const fetchFn = respondWith(200)
    await apiGet('/languages')
    await apiPost('/translate', BODY)
    expect(sent(fetchFn, 0).headers).not.toHaveProperty(PURPOSE_HEADER)
    expect(sent(fetchFn, 1).headers).not.toHaveProperty(PURPOSE_HEADER)
  })

  it('sends the purpose stated for translator on GET and POST alike', async () => {
    setPurpose(PURPOSE_SCOPE, { caseRef: 'case-0001', reason: 'Quality review' })
    const fetchFn = respondWith(200)
    await apiGet('/languages')
    await apiPost('/translate', BODY)
    const encoded = 'case_ref=case-0001&reason=Quality+review'
    expect(sent(fetchFn, 0).headers).toMatchObject({ [PURPOSE_HEADER]: encoded })
    expect(sent(fetchFn, 1).headers).toEqual({
      'content-type': 'application/json',
      [PURPOSE_HEADER]: encoded,
    })
  })

  it('ignores a purpose stated in another app of the same origin', async () => {
    setPurpose('docint', { caseRef: 'case-0002' })
    const fetchFn = respondWith(200)
    await apiPost('/translate', BODY)
    expect(sent(fetchFn, 0).headers).not.toHaveProperty(PURPOSE_HEADER)
  })
})

describe('a 428 asks for a purpose', () => {
  it('opens the purpose bar and sends once more with the purpose saved there', async () => {
    render(<PurposeBar scope={PURPOSE_SCOPE} />)
    const fetchFn = respondWith(428, 200)
    const user = userEvent.setup()

    const pending = apiPost('/translate', BODY)
    const dialog = await screen.findByRole('dialog', { name: 'Purpose' })
    expect(dialog).toHaveTextContent('This action needs a stated purpose.')
    await user.type(screen.getByLabelText('Case reference'), 'case-0001')
    await user.click(screen.getByRole('button', { name: 'Save' }))

    await expect(pending).resolves.toEqual({ ok: true })
    expect(fetchFn).toHaveBeenCalledTimes(2)
    expect(sent(fetchFn, 0).headers).not.toHaveProperty(PURPOSE_HEADER)
    expect(sent(fetchFn, 1).headers).toMatchObject({ [PURPOSE_HEADER]: 'case_ref=case-0001' })
    expect(sent(fetchFn, 1).body).toBe(sent(fetchFn, 0).body)
  })

  it('hands the 428 back as an ApiError when the user cancels', async () => {
    render(<PurposeBar scope={PURPOSE_SCOPE} required />)
    const fetchFn = respondWith(428)
    const user = userEvent.setup()

    // Caught up front: it rejects as soon as Cancel is clicked.
    const outcome = apiPost('/translate', BODY).catch((e: unknown) => e)
    await screen.findByRole('dialog', { name: 'Purpose' })
    await user.click(screen.getByRole('button', { name: 'Cancel' }))

    const error = await outcome
    expect(error).toBeInstanceOf(ApiError)
    expect((error as ApiError).status).toBe(428)
    expect(fetchFn).toHaveBeenCalledTimes(1)
  })

  it('fails at once when no purpose bar is mounted to ask with', async () => {
    const fetchFn = respondWith(428)
    const error = await apiPost('/translate', BODY).catch((e: unknown) => e)
    expect(error).toBeInstanceOf(ApiError)
    expect((error as ApiError).status).toBe(428)
    expect(fetchFn).toHaveBeenCalledTimes(1)
  })
})
