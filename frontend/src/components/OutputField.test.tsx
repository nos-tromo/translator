import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import { OutputField } from './OutputField'

describe('OutputField', () => {
  it('shows the placeholder while empty', () => {
    render(<OutputField value="" placeholder="Translation will appear here." />)
    expect(screen.getByText('Translation will appear here.')).toBeInTheDocument()
  })

  it('renders only the output, no controls', () => {
    const { container } = render(<OutputField value="Bonjour le monde" />)
    expect(screen.getByText('Bonjour le monde')).toBeInTheDocument()
    expect(container.querySelector('button')).toBeNull()
  })
})
