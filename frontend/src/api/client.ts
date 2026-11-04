import { withPurpose } from '@infra/ui'

/** API base derived from Vite's base path so the SPA works under its
 *  canonical sub-path (/translator/) and at root (dev) alike. */
export function apiBase(base: string = import.meta.env.BASE_URL): string {
  return `${base.replace(/\/+$/, '')}/api/v1`
}

export const API_BASE = apiBase()

/** The purpose store's scope, translator's compose project name. The
 *  header's PurposeBar and every request below use the same string. */
export const PURPOSE_SCOPE = 'translator'

export class ApiError extends Error {
  readonly status: number
  readonly detail: unknown
  constructor(status: number, detail: unknown) {
    super(`API ${status}`)
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

type SendInit = Omit<RequestInit, 'headers'> & { headers?: Record<string, string> }

export interface PostOptions {
  signal?: AbortSignal
  /** Abort an attempt whose response has not arrived within this many ms.
   *  The limit is per attempt: the time a 428 waits for the user to state a
   *  purpose counts against neither the first attempt nor the resend. */
  timeoutMs?: number
}

/** The one place a request leaves the SPA. It carries the purpose stated in
 *  the header's PurposeBar (X-Audit-Purpose); a 428 opens the bar's dialog
 *  and sends the request once more with the purpose saved there, or hands
 *  the 428 back when the user cancels. The body is a string, so it can be
 *  sent twice. */
function send(path: string, init: SendInit = {}, timeoutMs?: number): Promise<Response> {
  return withPurpose(PURPOSE_SCOPE, (purpose) =>
    attempt(`${API_BASE}${path}`, { ...init, headers: { ...init.headers, ...purpose } }, timeoutMs),
  )
}

/** One attempt of a request, with its own time limit when one is given.
 *  The caller's signal still aborts it. */
async function attempt(url: string, init: SendInit, timeoutMs?: number): Promise<Response> {
  if (timeoutMs === undefined) return fetch(url, init)
  const controller = new AbortController()
  const abort = () => controller.abort()
  const outer = init.signal
  if (outer?.aborted) abort()
  else outer?.addEventListener('abort', abort, { once: true })
  const timer = setTimeout(abort, timeoutMs)
  try {
    return await fetch(url, { ...init, signal: controller.signal })
  } finally {
    clearTimeout(timer)
    outer?.removeEventListener('abort', abort)
  }
}

export async function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  return parse<T>(await send(path, { signal }))
}

export async function apiPost<T>(path: string, json: unknown, options: PostOptions = {}): Promise<T> {
  const res = await send(
    path,
    {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(json),
      signal: options.signal,
    },
    options.timeoutMs,
  )
  return parse<T>(res)
}
