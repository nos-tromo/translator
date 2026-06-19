import { useQuery } from '@tanstack/react-query'
import { getLanguages } from '../api/translator'
import type { Language } from '../api/types'

/** Load the supported languages once, sorted by display name. */
export function useLanguages() {
  return useQuery({
    queryKey: ['languages'],
    queryFn: ({ signal }) => getLanguages(signal),
    staleTime: Infinity,
    select: (langs: Language[]) => [...langs].sort((a, b) => a.name.localeCompare(b.name)),
  })
}
