import { QueryClientProvider, useQuery } from '@tanstack/react-query'
import { AppShell, PageHeader } from '@infra/ui'
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
    <AppShell
      title="translator"
      version={data?.version ? `v${data.version}` : undefined}
      user={whoami?.display_name ?? whoami?.username ?? undefined}
      homeLabel={t('appheader.home')}
      themeLabels={{
        system: t('appheader.theme_system'),
        light: t('appheader.theme_light'),
        dark: t('appheader.theme_dark'),
      }}
      signOutLabel={t('appheader.sign_out')}
    >
      <div className="flex h-full min-h-0 flex-col p-8">
        <PageHeader title={t('page.title')} caption={t('page.caption')} />
        <TranslatePanel />
      </div>
    </AppShell>
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
