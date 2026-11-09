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

  it('loads the same file twice', async () => {
    const onText = vi.fn()
    const { container } = render(<FileTextInput onText={onText} onError={vi.fn()} />)
    const file = new File(['Hallo Welt'], 'notiz.txt', { type: 'text/plain' })
    fireEvent.change(fileInput(container), { target: { files: [file] } })
    await waitFor(() => expect(onText).toHaveBeenCalledTimes(1))
    fireEvent.change(fileInput(container), { target: { files: [file] } })
    await waitFor(() => expect(onText).toHaveBeenCalledTimes(2))
    expect(onText).toHaveBeenLastCalledWith('Hallo Welt')
  })

  it('draws no typed symbol', () => {
    const { container } = render(<FileTextInput onText={vi.fn()} onError={vi.fn()} />)
    expect(container.textContent).toBe('Upload .txt')
  })
})
