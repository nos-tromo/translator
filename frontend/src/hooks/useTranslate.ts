import { useMutation } from '@tanstack/react-query'
import { translate } from '../api/translator'
import type { TranslateRequest, TranslateResponse } from '../api/types'

/** Mutation wrapper around POST /translate. */
export function useTranslate() {
  return useMutation<TranslateResponse, Error, TranslateRequest>({
    mutationFn: (req) => translate(req),
  })
}
