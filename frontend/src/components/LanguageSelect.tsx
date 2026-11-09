import type { ReactNode } from 'react'
import { SectionHeader, SelectMenu } from '@infra/ui'
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
  /** A quiet action for the column this picker heads, on the caption's row. */
  action?: ReactNode
}

export function LanguageSelect({
  id,
  label,
  value,
  onChange,
  languages,
  includeAuto = false,
  action,
}: LanguageSelectProps) {
  const t = useT()
  const options = [
    ...(includeAuto ? [{ value: AUTO_DETECT, label: t('select.auto_detect') }] : []),
    ...languages.map((lang) => ({ value: lang.code, label: lang.name })),
  ]
  return (
    // A <label> can no longer wrap this — the picker is a button, not a form
    // control — so the caption is a plain span and `label` carries the
    // accessible name. The trigger shows a language, which is a value and
    // cannot name the control.
    <div id={id} className="flex flex-col gap-1 text-sm text-muted-foreground">
      <SectionHeader caption={label} actions={action} />
      <SelectMenu
        variant="field"
        label={label}
        options={options}
        value={value}
        onChange={onChange}
      />
    </div>
  )
}
