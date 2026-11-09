import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AUTO_DETECT, LanguageSelect } from './LanguageSelect'

const LANGS = [
  { code: 'en', name: 'English' },
  { code: 'fr', name: 'French' },
]

describe('LanguageSelect', () => {
  it('names itself and shows the current language while closed', () => {
    render(
      <LanguageSelect id="target" label="Target language" value="en"
        onChange={() => {}} languages={LANGS} />,
    )
    // The caption stays on screen beside the control, which the picker itself
    // no longer carries: its trigger shows a value, not a name.
    expect(screen.getByText('Target language')).toBeInTheDocument()
    expect(screen.getByRole('combobox', { name: 'Target language' })).toHaveTextContent('English')
  })

  it('renders an Auto-detect option when includeAuto is set', async () => {
    render(
      <LanguageSelect id="source" label="Source language" value={AUTO_DETECT}
        onChange={() => {}} languages={LANGS} includeAuto />,
    )
    await userEvent.click(screen.getByRole('combobox', { name: 'Source language' }))
    expect(screen.getByRole('option', { name: 'Auto-detect' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'French' })).toBeInTheDocument()
  })

  it('omits Auto-detect by default and reports selection by code', async () => {
    const onChange = vi.fn()
    render(
      <LanguageSelect id="target" label="Target language" value="en"
        onChange={onChange} languages={LANGS} />,
    )
    await userEvent.click(screen.getByRole('combobox', { name: 'Target language' }))
    expect(screen.queryByRole('option', { name: 'Auto-detect' })).toBeNull()
    await userEvent.click(screen.getByRole('option', { name: 'French' }))
    expect(onChange).toHaveBeenCalledWith('fr')
  })

  it('finds a language by typing its first letters', async () => {
    const onChange = vi.fn()
    render(
      <LanguageSelect id="target" label="Target language" value="en"
        onChange={onChange} languages={LANGS} />,
    )
    // The one thing a native <select> gave the operator for free, kept.
    screen.getByRole('combobox', { name: 'Target language' }).focus()
    await userEvent.keyboard('f')
    expect(onChange).toHaveBeenCalledWith('fr')
  })

  it('puts an action on its caption row', () => {
    render(
      <LanguageSelect id="target" label="Target language" value="en"
        onChange={() => {}} languages={LANGS} action={<button>Copy</button>} />,
    )
    expect(screen.getByRole('button', { name: 'Copy' }).closest('.flex-wrap')).toHaveTextContent('Target language')
  })

  it('gives its caption row one height with or without an action, so side-by-side pickers line up', () => {
    render(
      <>
        <LanguageSelect id="source" label="Source language" value="en" onChange={() => {}} languages={LANGS} />
        <LanguageSelect id="target" label="Target language" value="en" onChange={() => {}} languages={LANGS}
          action={<button>Copy</button>} />
      </>,
    )
    for (const caption of ['Source language', 'Target language']) {
      expect(screen.getByText(caption).closest('.flex-wrap')).toHaveClass('min-h-8')
    }
  })
})
