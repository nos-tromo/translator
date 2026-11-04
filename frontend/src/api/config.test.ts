import { describe, expect, it } from 'vitest'
import { purposeRequired, type AppConfig } from './config'

describe('purposeRequired', () => {
  it('follows the actions the backend reports', () => {
    expect(purposeRequired({ language: 'en', require_purpose: ['ai_call'] })).toBe(true)
    expect(purposeRequired({ language: 'en', require_purpose: [] })).toBe(false)
  })

  it('counts a config that is not loaded, or lacks the field, as not required', () => {
    expect(purposeRequired(undefined)).toBe(false)
    expect(purposeRequired({ language: 'de' } as AppConfig)).toBe(false)
  })
})
