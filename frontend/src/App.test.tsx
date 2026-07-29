import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import App from './App'
import { queryClient } from './api/queryClient'

vi.mock('./components/TranslatePanel', () => ({
  TranslatePanel: () => <div data-testid="translate-panel" />,
}))
vi.mock('./api/config', () => ({
  getConfig: vi.fn().mockResolvedValue({ language: 'en' }),
}))
const getWhoami = vi.fn()
vi.mock('./api/translator', () => ({
  getVersion: vi.fn().mockResolvedValue({ version: '1.2.3' }),
  getWhoami: () => getWhoami(),
}))

afterEach(() => {
  vi.restoreAllMocks()
  queryClient.clear()
})

describe('App', () => {
  it('renders exactly one header row, with the version shown in it', async () => {
    getWhoami.mockResolvedValue({ username: null, display_name: null })
    render(<App />)
    expect(await screen.findByTestId('appheader-version')).toHaveTextContent('v1.2.3')
    expect(screen.getAllByRole('banner')).toHaveLength(1)
    expect(await screen.findByTestId('translate-panel')).toBeInTheDocument()
  })

  it('shows the display name in the header when whoami resolves one', async () => {
    getWhoami.mockResolvedValue({ username: 'alex', display_name: 'Alex Example' })
    render(<App />)
    expect(await screen.findByTestId('appheader-user')).toHaveTextContent('Alex Example')
  })

  it('falls back to username when whoami has no display name', async () => {
    getWhoami.mockResolvedValue({ username: 'alex', display_name: null })
    render(<App />)
    expect(await screen.findByTestId('appheader-user')).toHaveTextContent('alex')
  })

  it('omits the user block when whoami has neither field', async () => {
    getWhoami.mockResolvedValue({ username: null, display_name: null })
    render(<App />)
    await screen.findByTestId('appheader-version')
    expect(screen.queryByTestId('appheader-user')).not.toBeInTheDocument()
  })
})
