import { Banner } from '@infra/ui'

interface DetectedLanguageBannerProps {
  name: string
  flag: string
  auto: boolean
}

/** Shows the detected (auto) or explicit source language with its flag. */
export function DetectedLanguageBanner({ name, flag, auto }: DetectedLanguageBannerProps) {
  const label = auto ? 'Detected source language' : 'Source language'
  return (
    <Banner variant="info">
      <span className="text-muted-foreground">{label}: </span>
      <span>
        {flag} {name}
      </span>
    </Banner>
  )
}
