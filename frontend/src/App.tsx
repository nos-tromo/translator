import { QueryClientProvider } from '@tanstack/react-query'
import { queryClient } from './api/queryClient'
import { TranslatePanel } from './components/TranslatePanel'

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <div className="min-h-full">
        <header className="border-b border-border px-6 py-4">
          <h1 className="text-lg font-semibold">🌐 Translator</h1>
        </header>
        <main className="mx-auto max-w-5xl px-6 py-8">
          <TranslatePanel />
        </main>
      </div>
    </QueryClientProvider>
  )
}
