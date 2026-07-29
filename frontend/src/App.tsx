import { QueryClientProvider } from '@tanstack/react-query'
import { AppHeader, Shell } from '@infra/ui'
import { queryClient } from './api/queryClient'
import { TranslatePanel } from './components/TranslatePanel'
import { VersionBadge } from './components/VersionBadge'
import { LanguageProvider, useT } from './i18n/LanguageContext'

function AppContent() {
  const t = useT()
  return (
    <>
      <AppHeader
        title="translator"
        homeLabel={t('appheader.home')}
        themeLabels={{
          system: t('appheader.theme_system'),
          light: t('appheader.theme_light'),
          dark: t('appheader.theme_dark'),
        }}
      />
      <Shell title="Translator" actions={<VersionBadge />}>
        <TranslatePanel />
      </Shell>
    </>
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
