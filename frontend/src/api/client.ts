/** API base derived from Vite's base path so the SPA works under its
 *  canonical sub-path (/translator/) and at root (dev) alike. */
export function apiBase(base: string = import.meta.env.BASE_URL): string {
  return `${base.replace(/\/+$/, '')}/api/v1`
}

export const API_BASE = apiBase()

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
