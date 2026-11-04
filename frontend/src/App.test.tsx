import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { clearAllPurposes } from '@infra/ui'
import App from './App'
import type { AppConfig } from './api/config'
import { queryClient } from './api/queryClient'

vi.mock('./components/TranslatePanel', () => ({
  TranslatePanel: () => <div data-testid="translate-panel" />,
}))
const getConfig = vi.fn<() => Promise<AppConfig>>()
vi.mock('./api/config', async (importOriginal) => ({
  ...(await importOriginal<typeof import('./api/config')>()),
  getConfig: () => getConfig(),
}))
const getWhoami = vi.fn()
vi.mock('./api/translator', () => ({
  getVersion: vi.fn().mockResolvedValue({ version: '1.2.3' }),
  getWhoami: () => getWhoami(),
}))

beforeEach(() => {
  getConfig.mockResolvedValue({ language: 'en', require_purpose: [] })
  getWhoami.mockResolvedValue({ username: null, display_name: null })
})

afterEach(() => {
  vi.restoreAllMocks()
  queryClient.clear()
  clearAllPurposes()
  sessionStorage.clear()
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

describe('the purpose bar', () => {
  it('sits in the header, optional while the backend requires no purpose', async () => {
    render(<App />)
    const bar = await screen.findByRole('button', { name: 'No purpose stated' })
    expect(bar).toHaveAttribute('data-state', 'empty')
    expect(within(screen.getByRole('banner')).getByRole('button', { name: 'No purpose stated' })).toBe(bar)
  })

  it('is required while the config lists an action that needs a purpose', async () => {
    getConfig.mockResolvedValue({ language: 'en', require_purpose: ['ai_call'] })
    render(<App />)
    const bar = await screen.findByRole('button', { name: 'Purpose required' })
    expect(bar).toHaveAttribute('data-state', 'required')
  })

  it('speaks German in a German UI', async () => {
    getConfig.mockResolvedValue({ language: 'de', require_purpose: ['ai_call'] })
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole('button', { name: 'Zweck erforderlich' }))

    const dialog = screen.getByRole('dialog', { name: 'Zweck' })
    expect(within(dialog).getByLabelText('Aktenzeichen')).toHaveFocus()
    expect(within(dialog).getByLabelText('Begründung')).toBeInTheDocument()
    await user.click(within(dialog).getByRole('button', { name: 'Speichern' }))
    expect(dialog).toHaveTextContent('Bitte geben Sie ein Aktenzeichen oder eine Begründung ein.')
    await user.type(within(dialog).getByLabelText('Aktenzeichen'), 'case-0001')
    await user.click(within(dialog).getByRole('button', { name: 'Speichern' }))

    expect(screen.queryByRole('dialog')).toBeNull()
    expect(screen.getByRole('button', { name: 'Zweck: case-0001' })).toHaveAttribute('data-state', 'set')
  })
})
