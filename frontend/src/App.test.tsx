import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import App from './App'

vi.mock('./components/TranslatePanel', () => ({
  TranslatePanel: () => <div data-testid="translate-panel" />,
}))
vi.mock('./api/config', () => ({
  getConfig: vi.fn().mockResolvedValue({ language: 'en' }),
}))
vi.mock('./api/translator', () => ({
  getVersion: vi.fn().mockResolvedValue({ version: '1.2.3' }),
}))

afterEach(() => vi.restoreAllMocks())

describe('App', () => {
  it('renders exactly one header row, with the version shown in it', async () => {
    render(<App />)
    expect(await screen.findByTestId('appheader-version')).toHaveTextContent('v1.2.3')
    expect(screen.getAllByRole('banner')).toHaveLength(1)
    expect(await screen.findByTestId('translate-panel')).toBeInTheDocument()
  })
})
