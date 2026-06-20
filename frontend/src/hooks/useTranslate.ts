import { useMutation } from '@tanstack/react-query'
import { translate } from '../api/translator'
import type { TranslateRequest, TranslateResponse } from '../api/types'

const TRANSLATE_TIMEOUT_MS = 120_000

/** Mutation wrapper around POST /translate with a client-side ~120s abort cap. */
export function useTranslate() {
  return useMutation<TranslateResponse, Error, TranslateRequest>({
    mutationFn: async (req) => {
      const controller = new AbortController()
      const timer = setTimeout(() => controller.abort(), TRANSLATE_TIMEOUT_MS)
      try {
        return await translate(req, controller.signal)
      } finally {
        clearTimeout(timer)
      }
    },
  })
}
