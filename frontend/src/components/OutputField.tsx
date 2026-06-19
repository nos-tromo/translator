import { useState } from 'react'
import { Button, cn } from '@infra/ui'

interface OutputFieldProps {
  value: string
  placeholder?: string
}

/** Read-only translation output with a copy button revealed on hover/focus. */
export function OutputField({ value, placeholder }: OutputFieldProps) {
  const [copied, setCopied] = useState(false)

  async function copy() {
    await navigator.clipboard.writeText(value)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1500)
  }

  return (
    <div className="group relative h-72 overflow-auto rounded-md border border-border bg-muted/30 p-3">
      {value ? (
        <p className="whitespace-pre-wrap text-sm text-foreground">{value}</p>
      ) : (
        <p className="text-sm text-muted-foreground">{placeholder}</p>
      )}
      {value && (
        <Button
          type="button"
          variant="secondary"
          size="sm"
          aria-label={copied ? 'Copied' : 'Copy translation'}
          onClick={copy}
          className={cn(
            'absolute right-2 top-2 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100',
            copied && 'opacity-100',
          )}
        >
          {copied ? '✓ Copied' : 'Copy'}
        </Button>
      )}
    </div>
  )
}
