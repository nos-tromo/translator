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
    expect(await screen.findByText('v1.2.3')).toBeInTheDocument()
    expect(screen.getAllByRole('banner')).toHaveLength(1)
    expect(await screen.findByTestId('translate-panel')).toBeInTheDocument()
  })

  it('shows the user menu with the display name when whoami resolves one', async () => {
    getWhoami.mockResolvedValue({ username: 'alex', display_name: 'Alex Example' })
    render(<App />)
    expect(await screen.findByRole('button', { name: /Alex Example/ })).toBeInTheDocument()
  })

  it('falls back to username in the user menu when whoami has no display name', async () => {
    getWhoami.mockResolvedValue({ username: 'alex', display_name: null })
    render(<App />)
    expect(await screen.findByRole('button', { name: /alex/ })).toBeInTheDocument()
  })

  it('omits the user menu when whoami has neither field', async () => {
    getWhoami.mockResolvedValue({ username: null, display_name: null })
    render(<App />)
    await screen.findByText('v1.2.3')
    expect(screen.queryByRole('button', { name: /account/i })).toBeNull()
  })
})
