import { apiGet, apiPost } from './client'
import type { Language, TranslateRequest, TranslateResponse } from './types'

export function getLanguages(signal?: AbortSignal): Promise<Language[]> {
  return apiGet<Language[]>('/languages', signal)
}

/** How long one attempt of a translation may take before it is aborted. A
 *  resend after the purpose prompt gets the whole limit again. */
export const TRANSLATE_TIMEOUT_MS = 120_000

export function translate(req: TranslateRequest, signal?: AbortSignal): Promise<TranslateResponse> {
  return apiPost<TranslateResponse>('/translate', req, { signal, timeoutMs: TRANSLATE_TIMEOUT_MS })
}

export function getVersion(): Promise<{ version: string }> {
  return apiGet<{ version: string }>('/version') // → GET /api/v1/version
}

export interface Whoami {
  username: string | null
  display_name: string | null
}

export function getWhoami(): Promise<Whoami> {
  return apiGet<Whoami>('/whoami') // → GET /api/v1/whoami
}
