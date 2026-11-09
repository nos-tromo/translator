interface OutputFieldProps {
  value: string
  placeholder?: string
}

/** Read-only translation output. Its copy action sits on the target column's header row. */
export function OutputField({ value, placeholder }: OutputFieldProps) {
  return (
    <div className="h-full min-h-72 overflow-auto rounded-lg border border-border bg-muted p-3">
      {value ? (
        <p className="whitespace-pre-wrap text-sm text-foreground">{value}</p>
      ) : (
        <p className="text-sm text-muted-foreground">{placeholder}</p>
      )}
    </div>
  )
}
