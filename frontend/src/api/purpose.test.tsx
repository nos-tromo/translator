import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { PURPOSE_HEADER, PurposeBar, clearAllPurposes, setPurpose } from '@infra/ui'
import { ApiError, PURPOSE_SCOPE, apiGet, apiPost } from './client'
import { TRANSLATE_TIMEOUT_MS, translate } from './translator'

const BODY = { text: 'Hello Alice Example', target_lang: 'fr' }

afterEach(() => {
  vi.useRealTimers()
  clearAllPurposes()
  sessionStorage.clear()
  vi.restoreAllMocks()
})

function json(status: number, body: unknown) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  })
}

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

describe('the translation time limit applies per attempt', () => {
  const RESULT = { translation: 'Bonjour Alice Example', detected_language: { name: 'English', flag: '' } }

  /** Stub fetch: the first calls answer at once with these statuses, every
   *  later one waits until it is answered or aborted. */
  function slowAfter(...statuses: number[]) {
    const signals: AbortSignal[] = []
    const answers: Array<(res: Response) => void> = []
    const fn = vi.fn((_input: RequestInfo | URL, init?: RequestInit) => {
      const signal = init!.signal!
      signals.push(signal)
      const status = statuses[signals.length - 1]
      if (status !== undefined) {
        return Promise.resolve(json(status, { detail: 'This action requires a stated purpose.' }))
      }
      return new Promise<Response>((resolve, reject) => {
        answers.push(resolve)
        signal.addEventListener('abort', () =>
          reject(new DOMException('The operation was aborted.', 'AbortError')),
        )
      })
    })
    vi.stubGlobal('fetch', fn)
    return { fn, signals, answers }
  }

  const elapse = (ms: number) =>
    act(async () => {
      await vi.advanceTimersByTimeAsync(ms)
    })

  it('aborts an attempt whose answer takes longer than the limit', async () => {
    vi.useFakeTimers()
    const { signals } = slowAfter()

    const outcome = translate(BODY).catch((e: unknown) => e)
    await elapse(TRANSLATE_TIMEOUT_MS - 1)
    expect(signals[0].aborted).toBe(false)
    await elapse(1)

    expect(signals[0].aborted).toBe(true)
    expect(await outcome).toMatchObject({ name: 'AbortError' })
  })

  it('gives the resend after a 428 the whole limit, however long the dialog took', async () => {
    vi.useFakeTimers()
    render(<PurposeBar scope={PURPOSE_SCOPE} />)
    const { fn, signals, answers } = slowAfter(428)

    const outcome = translate(BODY)
    await elapse(0)
    expect(screen.getByRole('dialog')).toBeInTheDocument()
    // Longer in the dialog than the whole limit: no attempt is running.
    await elapse(TRANSLATE_TIMEOUT_MS + 30_000)
    expect(signals[0].aborted).toBe(false)
    expect(fn).toHaveBeenCalledTimes(1)

    // What the bar's Save does, then the resend.
    act(() => {
      setPurpose(PURPOSE_SCOPE, { caseRef: 'case-0001' })
    })
    await elapse(0)
    expect(fn).toHaveBeenCalledTimes(2)
    await elapse(TRANSLATE_TIMEOUT_MS - 1)
    expect(signals[1].aborted).toBe(false)
    answers[0](json(200, RESULT))

    await expect(outcome).resolves.toEqual(RESULT)
  })

  it('still lets the caller abort an attempt', async () => {
    const { signals } = slowAfter()
    const controller = new AbortController()

    const outcome = translate(BODY, controller.signal).catch((e: unknown) => e)
    controller.abort()

    expect(signals[0].aborted).toBe(true)
    expect(await outcome).toMatchObject({ name: 'AbortError' })
  })
})
