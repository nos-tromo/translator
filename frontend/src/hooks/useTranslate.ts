import { useMutation } from '@tanstack/react-query'
import { translate } from '../api/translator'
import type { TranslateRequest, TranslateResponse } from '../api/types'

/** Mutation wrapper around POST /translate. Its ~120 s abort cap applies to
 *  each attempt (api/translator.ts), so time spent stating a purpose after a
 *  428 does not shorten the resend. */
export function useTranslate() {
  return useMutation<TranslateResponse, Error, TranslateRequest>({
    mutationFn: (req) => translate(req),
  })
}
