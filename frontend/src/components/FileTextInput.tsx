import type { ChangeEvent } from 'react'
import { useT } from '../i18n/LanguageContext'

interface FileTextInputProps {
  onText: (text: string) => void
  onError: (message: string) => void
}

/** A `.txt` picker that reads the file client-side and emits its UTF-8 text. */
export function FileTextInput({ onText, onError }: FileTextInputProps) {
  const t = useT()
  function handleChange(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    const reader = new FileReader()
    reader.onload = () => onText(typeof reader.result === 'string' ? reader.result : '')
    reader.onerror = () => onError(t('fileinput.read_error'))
    reader.readAsText(file, 'utf-8')
    e.target.value = '' // allow re-selecting the same file
  }
  return (
    <label className="inline-flex cursor-pointer items-center gap-2 text-sm text-muted-foreground hover:text-foreground">
      <span aria-hidden>📎</span>
      {t('fileinput.upload_txt')}
      <input type="file" accept=".txt,text/plain" className="sr-only" onChange={handleChange} />
    </label>
  )
}
