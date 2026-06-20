import { Select } from '@infra/ui'
import type { Language } from '../api/types'

/** Sentinel value for the source selector's "Auto-detect" choice. */
export const AUTO_DETECT = '__auto__'

interface LanguageSelectProps {
  id: string
  label: string
  value: string
  onChange: (value: string) => void
  languages: Language[]
  includeAuto?: boolean
}

export function LanguageSelect({
  id,
  label,
  value,
  onChange,
  languages,
  includeAuto = false,
}: LanguageSelectProps) {
  return (
    <label htmlFor={id} className="flex flex-col gap-1 text-sm text-muted-foreground">
      {label}
      <Select id={id} value={value} onChange={(e) => onChange(e.target.value)}>
        {includeAuto && <option value={AUTO_DETECT}>Auto-detect</option>}
        {languages.map((lang) => (
          <option key={lang.code} value={lang.code}>
            {lang.name}
          </option>
        ))}
      </Select>
    </label>
  )
}
