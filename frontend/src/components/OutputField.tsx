import { CopyButton } from '@infra/ui'

interface OutputFieldProps {
  value: string
  placeholder?: string
}

/** Read-only translation output with a copy button revealed on hover/focus. */
export function OutputField({ value, placeholder }: OutputFieldProps) {
  return (
    <div className="group relative h-72 overflow-auto rounded-md border border-border bg-muted/30 p-3">
      {value ? (
        <p className="whitespace-pre-wrap text-sm text-foreground">{value}</p>
      ) : (
        <p className="text-sm text-muted-foreground">{placeholder}</p>
      )}
      {value && (
        <CopyButton
          text={value}
          label="Copy translation"
          variant="secondary"
          className="absolute right-2 top-2 opacity-0 transition-opacity focus:opacity-100 group-hover:opacity-100"
        />
      )}
    </div>
  )
}
