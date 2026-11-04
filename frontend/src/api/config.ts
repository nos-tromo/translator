import { apiGet } from './client'

export interface AppConfig {
  language: 'en' | 'de'
  /** The audit actions of translator's endpoints that need a stated purpose. */
  require_purpose: string[]
}

export const getConfig = () => apiGet<AppConfig>('/config')

/** Whether the backend refuses some of translator's requests without a
 *  stated purpose. Unknown (config not loaded) counts as not required: the
 *  backend still answers 428 then, and the purpose bar asks. */
export function purposeRequired(config: AppConfig | undefined): boolean {
  return (config?.require_purpose?.length ?? 0) > 0
}
