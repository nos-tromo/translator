import { apiGet, apiPost } from './client'
import type { Language, TranslateRequest, TranslateResponse } from './types'

export function getLanguages(signal?: AbortSignal): Promise<Language[]> {
  return apiGet<Language[]>('/languages', signal)
}

export function translate(req: TranslateRequest, signal?: AbortSignal): Promise<TranslateResponse> {
  return apiPost<TranslateResponse>('/translate', req, signal)
}
