import { useRef, type ChangeEvent } from 'react'
import { Button } from '@infra/ui'
import { useT } from '../i18n/LanguageContext'

interface FileTextInputProps {
  onText: (text: string) => void
  onError: (message: string) => void
}

/**
 * A `.txt` picker that reads the file client-side and emits its UTF-8 text.
 *
 * A real button opening a hidden file input, so it sits in the panel's action
 * bar like any other control and is reachable by keyboard.
 */
export function FileTextInput({ onText, onError }: FileTextInputProps) {
  const t = useT()
  const inputRef = useRef<HTMLInputElement>(null)
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
    <>
      <Button type="button" variant="secondary" onClick={() => inputRef.current?.click()}>
        {t('fileinput.upload_txt')}
      </Button>
      <input
        ref={inputRef}
        type="file"
        accept=".txt,text/plain"
        className="hidden"
        tabIndex={-1}
        aria-hidden
        onChange={handleChange}
      />
    </>
  )
}
