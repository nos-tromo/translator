import { QueryClientProvider } from '@tanstack/react-query'
import { Shell } from '@infra/ui'
import { queryClient } from './api/queryClient'
import { TranslatePanel } from './components/TranslatePanel'
import { VersionBadge } from './components/VersionBadge'

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Shell title="Translator" actions={<VersionBadge />}>
        <TranslatePanel />
      </Shell>
    </QueryClientProvider>
  )
}
