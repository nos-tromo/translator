import { useQuery } from '@tanstack/react-query'
import { getConfig } from '../api/config'

/** The SPA bootstrap config, fetched once per page load and shared by every
 *  caller: the UI language and whether a purpose is required. */
export function useAppConfig() {
  return useQuery({
    queryKey: ['app-config'],
    queryFn: getConfig,
    staleTime: Infinity,
    gcTime: Infinity,
    retry: 1,
  })
}
