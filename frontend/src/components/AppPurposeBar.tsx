import { PurposeBar, type PurposeBarLabels } from '@infra/ui'
import { PURPOSE_SCOPE } from '../api/client'
import { useT } from '../i18n/LanguageContext'

interface AppPurposeBarProps {
  /** The backend refuses a translation without a stated purpose. */
  required: boolean
}

/** The header's PurposeBar: where the user states the purpose every API
 *  request carries (api/client.ts), in the UI language. */
export function AppPurposeBar({ required }: AppPurposeBarProps) {
  const t = useT()
  // Every label, typed in full rather than Partial: one that a newer
  // @infra/ui adds fails the build until it is translated here.
  const labels: PurposeBarLabels = {
    purpose: t('purpose.label'),
    empty: t('purpose.empty'),
    required: t('purpose.required'),
    title: t('purpose.title'),
    description: t('purpose.description'),
    requested: t('purpose.requested'),
    caseRef: t('purpose.case_ref'),
    reason: t('purpose.reason'),
    save: t('purpose.save'),
    cancel: t('purpose.cancel'),
    clear: t('purpose.clear'),
    errorLength: t('purpose.error_length'),
    errorControl: t('purpose.error_control'),
    errorHeader: t('purpose.error_header'),
    errorEmpty: t('purpose.error_empty'),
  }
  return <PurposeBar scope={PURPOSE_SCOPE} required={required} labels={labels} />
}
