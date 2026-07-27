import { Select } from '@infra/ui'
import { useT } from '../i18n/LanguageContext'
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
  const t = useT()
  return (
    <label htmlFor={id} className="flex flex-col gap-1 text-sm text-muted-foreground">
      {label}
      <Select id={id} value={value} onChange={(e) => onChange(e.target.value)}>
        {includeAuto && <option value={AUTO_DETECT}>{t('select.auto_detect')}</option>}
        {languages.map((lang) => (
          <option key={lang.code} value={lang.code}>
            {lang.name}
          </option>
        ))}
      </Select>
    </label>
  )
}
