import { Banner } from '@infra/ui'
import { useT } from '../i18n/LanguageContext'

interface DetectedLanguageBannerProps {
  name: string
  flag: string
  auto: boolean
}

/** Shows the detected (auto) or explicit source language with its flag. */
export function DetectedLanguageBanner({ name, flag, auto }: DetectedLanguageBannerProps) {
  const t = useT()
  const label = auto ? t('banner.detected_source_language') : t('banner.source_language')
  return (
    <Banner variant="info">
      <span className="text-muted-foreground">{label}: </span>
      <span>
        {flag} {name}
      </span>
    </Banner>
  )
}
