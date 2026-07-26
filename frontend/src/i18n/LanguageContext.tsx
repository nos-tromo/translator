import { createContext, useContext } from 'react'
import { useQuery } from '@tanstack/react-query'
import { Spinner } from '@infra/ui'
import { getConfig } from '../api/config'
import { catalogs, format } from './index'
import type { Lang, Strings } from './index'

// Default 'en' means components (and their tests) work without a provider.
const LanguageContext = createContext<Lang>('en')

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const { data, isLoading } = useQuery({
    queryKey: ['app-config'],
    queryFn: getConfig,
    staleTime: Infinity,
    gcTime: Infinity,
    retry: 1,
  })
  // Block only the very first paint so a de deployment never flashes English;
  // a failed config fetch falls through to 'en' — it can never blank the UI.
  if (isLoading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Spinner label="…" />
      </div>
    )
  }
  const language: Lang = data?.language === 'de' ? 'de' : 'en'
  return <LanguageContext.Provider value={language}>{children}</LanguageContext.Provider>
}

export function useLang(): Lang {
  return useContext(LanguageContext)
}

export function useT(): (
  key: keyof Strings,
  vars?: Record<string, string | number>,
) => string {
  const lang = useContext(LanguageContext)
  return (key, vars) => format(catalogs[lang][key], vars)
}

export { LanguageContext }
