import { useMemo, useState } from 'react'
import { Banner, Button, Spinner } from '@infra/ui'
import { describeError } from '../api/errorMessage'
import { useLanguages } from '../hooks/useLanguages'
import { useTranslate } from '../hooks/useTranslate'
import { useT } from '../i18n/LanguageContext'
import { AUTO_DETECT, LanguageSelect } from './LanguageSelect'
import { FileTextInput } from './FileTextInput'
import { OutputField } from './OutputField'
import { DetectedLanguageBanner } from './DetectedLanguageBanner'

const DEFAULT_TARGET = import.meta.env.VITE_DEFAULT_TARGET_LANGUAGE ?? 'English'

export function TranslatePanel() {
  const t = useT()
  const languagesQuery = useLanguages()
  const translation = useTranslate()

  const [source, setSource] = useState(AUTO_DETECT)
  // Explicit user-chosen target (empty = not yet chosen; fall back to derived default)
  const [targetOverride, setTargetOverride] = useState('')
  const [text, setText] = useState('')
  const [fileError, setFileError] = useState<string | null>(null)

  const languages = useMemo(() => languagesQuery.data ?? [], [languagesQuery.data])

  // Derive the effective target: user choice, or the language matching DEFAULT_TARGET, or first.
  const target = useMemo(() => {
    if (targetOverride !== '') return targetOverride
    if (languages.length === 0) return ''
    const def = languages.find((l) => l.name === DEFAULT_TARGET) ?? languages[0]
    return def.code
  }, [targetOverride, languages])

  function onTranslate() {
    setFileError(null)
    translation.mutate({
      text: text.trim(),
      target_lang: target,
      source_lang: source === AUTO_DETECT ? undefined : source,
    })
  }

  if (languagesQuery.isLoading) return <Spinner label={t('panel.loading_languages')} />
  if (languagesQuery.error) {
    const d = describeError(languagesQuery.error)
    return <Banner variant="danger">{t(d.key, d.vars)}</Banner>
  }

  const translateError = translation.error ? describeError(translation.error) : null
  const result = translation.data
  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <LanguageSelect
          id="source"
          label={t('panel.source_language')}
          value={source}
          onChange={setSource}
          languages={languages}
          includeAuto
        />
        <LanguageSelect
          id="target"
          label={t('panel.target_language')}
          value={target}
          onChange={setTargetOverride}
          languages={languages}
        />
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-2">
          <textarea
            aria-label={t('panel.text_aria')}
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={t('panel.text_placeholder')}
            className="h-72 w-full resize-none rounded-md border border-border bg-background p-3 text-sm text-foreground placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
          />
          <FileTextInput onText={setText} onError={setFileError} />
        </div>
        <OutputField value={result?.translation ?? ''} placeholder={t('panel.output_placeholder')} />
      </div>

      <div className="flex items-center gap-3">
        <Button
          type="button"
          onClick={onTranslate}
          disabled={text.trim().length === 0 || target === '' || translation.isPending}
        >
          {t('panel.translate')}
        </Button>
        {translation.isPending && <Spinner label={t('panel.translating')} />}
      </div>

      {fileError && <Banner variant="danger">{fileError}</Banner>}
      {translateError && <Banner variant="danger">{t(translateError.key, translateError.vars)}</Banner>}
      {result && (
        <DetectedLanguageBanner
          name={result.detected_language.name}
          flag={result.detected_language.flag}
          auto={source === AUTO_DETECT}
        />
      )}
    </div>
  )
}
