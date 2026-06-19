import { afterEach, describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { OutputField } from './OutputField'

afterEach(() => vi.restoreAllMocks())

describe('OutputField', () => {
  it('shows the placeholder and no copy button when empty', () => {
    render(<OutputField value="" placeholder="Translation will appear here." />)
    expect(screen.getByText('Translation will appear here.')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /copy/i })).toBeNull()
  })

  it('copies the value to the clipboard when the button is clicked', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal('navigator', { clipboard: { writeText } })
    render(<OutputField value="Bonjour le monde" />)
    await userEvent.click(screen.getByRole('button', { name: /copy translation/i }))
    expect(writeText).toHaveBeenCalledWith('Bonjour le monde')
    expect(screen.getByRole('button', { name: /copied/i })).toBeInTheDocument()
  })
})
