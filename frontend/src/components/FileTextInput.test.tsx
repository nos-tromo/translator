import { afterEach, describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { FileTextInput } from './FileTextInput'

afterEach(() => vi.restoreAllMocks())

function fileInput(container: HTMLElement): HTMLInputElement {
  return container.querySelector('input[type="file"]') as HTMLInputElement
}

describe('FileTextInput', () => {
  it('is a real button that opens the picker', async () => {
    const click = vi.spyOn(HTMLInputElement.prototype, 'click').mockImplementation(() => {})
    render(<FileTextInput onText={vi.fn()} onError={vi.fn()} />)
    await userEvent.click(screen.getByRole('button', { name: 'Upload .txt' }))
    expect(click).toHaveBeenCalledOnce()
  })

  it('clears its value after each pick, so the same file can be picked again', async () => {
    const onText = vi.fn()
    const { container } = render(<FileTextInput onText={onText} onError={vi.fn()} />)
    const input = fileInput(container)
    const set = vi.spyOn(input, 'value', 'set')
    fireEvent.change(input, { target: { files: [new File(['Hallo Welt'], 'notiz.txt', { type: 'text/plain' })] } })
    await waitFor(() => expect(onText).toHaveBeenCalledWith('Hallo Welt'))
    expect(set).toHaveBeenCalledWith('')
  })

  it('draws no typed symbol', () => {
    const { container } = render(<FileTextInput onText={vi.fn()} onError={vi.fn()} />)
    expect(container.textContent).toBe('Upload .txt')
  })
})
