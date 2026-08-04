import { CopyButton } from '@infra/ui'
import { useT } from '../i18n/LanguageContext'

interface OutputFieldProps {
  value: string
  placeholder?: string
}

/** Read-only translation output with a copy button revealed on hover/focus. */
export function OutputField({ value, placeholder }: OutputFieldProps) {
  const t = useT()
  return (
    <div className="group relative h-full min-h-72 overflow-auto rounded-lg border border-border bg-muted p-3">
      {value ? (
        <p className="whitespace-pre-wrap text-sm text-foreground">{value}</p>
      ) : (
        <p className="text-sm text-muted-foreground">{placeholder}</p>
      )}
      {value && (
        <CopyButton
          text={value}
          label={t('output.copy_translation')}
          copiedLabel={t('common.copied')}
          variant="secondary"
          className="absolute right-2 top-2 opacity-0 transition-opacity focus:opacity-100 group-hover:opacity-100"
        />
      )}
    </div>
  )
}
