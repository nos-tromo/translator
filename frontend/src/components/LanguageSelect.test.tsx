import { describe, expect, it, vi } from 'vitest'
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AUTO_DETECT, LanguageSelect } from './LanguageSelect'

const LANGS = [
  { code: 'en', name: 'English' },
  { code: 'fr', name: 'French' },
]

describe('LanguageSelect', () => {
  it('renders an Auto-detect option when includeAuto is set', () => {
    render(
      <LanguageSelect id="source" label="Source language" value={AUTO_DETECT}
        onChange={() => {}} languages={LANGS} includeAuto />,
    )
    expect(screen.getByRole('option', { name: 'Auto-detect' })).toBeInTheDocument()
    expect(screen.getByRole('option', { name: 'French' })).toBeInTheDocument()
  })

  it('omits Auto-detect by default and reports selection by code', async () => {
    const onChange = vi.fn()
    render(
      <LanguageSelect id="target" label="Target language" value="en"
        onChange={onChange} languages={LANGS} />,
    )
    expect(screen.queryByRole('option', { name: 'Auto-detect' })).toBeNull()
    await userEvent.selectOptions(screen.getByLabelText('Target language'), 'fr')
    expect(onChange).toHaveBeenCalledWith('fr')
  })
})
