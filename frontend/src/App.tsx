import { QueryClientProvider, useQuery } from '@tanstack/react-query'
import { AppHeader } from '@infra/ui'
import { getVersion, getWhoami } from './api/translator'
import { queryClient } from './api/queryClient'
import { TranslatePanel } from './components/TranslatePanel'
import { LanguageProvider, useT } from './i18n/LanguageContext'

function AppContent() {
  const t = useT()
  const { data } = useQuery({
    queryKey: ['version'],
    queryFn: getVersion,
    staleTime: Infinity,
  })
  // Loading and error states both resolve to `undefined` so the header
  // simply omits the user block rather than showing a stale/error placeholder.
  const { data: whoami } = useQuery({
    queryKey: ['whoami'],
    queryFn: getWhoami,
    staleTime: Infinity,
    gcTime: Infinity,
    retry: false,
  })
  return (
    <div className="min-h-full">
      <AppHeader
        title="translator"
        version={data?.version ? `v${data.version}` : undefined}
        user={whoami?.display_name ?? whoami?.username ?? undefined}
        homeLabel={t('appheader.home')}
        themeLabels={{
          system: t('appheader.theme_system'),
          light: t('appheader.theme_light'),
          dark: t('appheader.theme_dark'),
        }}
      />
      <main className="mx-auto max-w-5xl px-6 py-8">
        <TranslatePanel />
      </main>
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <LanguageProvider>
        <AppContent />
      </LanguageProvider>
    </QueryClientProvider>
  )
}
