import type { Strings } from './index'

export const de: Strings = {
  // appheader
  'appheader.home': 'Übersicht',
  'appheader.theme_system': 'System',
  'appheader.theme_light': 'Hell',
  'appheader.theme_dark': 'Dunkel',
  'appheader.sign_out': 'Abmelden',

  // purpose (PurposeBar; {max} is filled in by the bar)
  'purpose.label': 'Zweck',
  'purpose.empty': 'Kein Zweck angegeben',
  'purpose.required': 'Zweck erforderlich',
  'purpose.title': 'Zweck',
  'purpose.description':
    'Wird mit jeder Ihrer Anfragen in dieser App gesendet, bis Sie ihn ändern oder entfernen oder die Sitzung endet.',
  'purpose.requested': 'Für diese Aktion muss ein Zweck angegeben werden.',
  'purpose.case_ref': 'Aktenzeichen',
  'purpose.reason': 'Begründung',
  'purpose.save': 'Speichern',
  'purpose.cancel': 'Abbrechen',
  'purpose.clear': 'Entfernen',
  'purpose.error_length': 'Höchstens {max} Zeichen.',
  'purpose.error_control': 'Bitte entfernen Sie Zeilenumbrüche, Tabulatoren und andere Steuerzeichen.',
  'purpose.error_header': 'Zusammen zu lang zum Senden. Bitte kürzen Sie eines der Felder.',
  'purpose.error_empty': 'Bitte geben Sie ein Aktenzeichen oder eine Begründung an.',

  // page
  'page.title': 'Übersetzen',
  'page.caption': 'Textübersetzung',

  // common
  'common.copied': 'Kopiert',
  'common.error_request': 'Etwas ist schiefgelaufen ({status}) – Bitte versuchen Sie es erneut oder kontaktieren Sie den Support.',
  'common.error_unknown': 'Etwas ist schiefgelaufen – Bitte versuchen Sie es erneut oder kontaktieren Sie den Support.',
  'common.error_network': 'Dienst nicht erreichbar – Bitte prüfen Sie Ihre Verbindung oder versuchen Sie es später erneut.',

  // panel
  'panel.source_language': 'Ausgangssprache',
  'panel.target_language': 'Zielsprache',
  'panel.text_aria': 'Zu übersetzender Text',
  'panel.text_placeholder': 'Zu übersetzenden Text eingeben…',
  'panel.output_placeholder': 'Die Übersetzung wird hier angezeigt.',
  'panel.translate': 'Übersetzen',
  'panel.translating': 'Wird übersetzt…',
  'panel.loading_languages': 'Sprachen werden geladen…',

  // select
  'select.auto_detect': 'Automatisch erkennen',

  // output
  'output.copy_translation': 'Übersetzung kopieren',

  // fileinput
  'fileinput.upload_txt': '.txt hochladen',
  'fileinput.read_error': 'Die Datei konnte nicht als UTF-8-Text gelesen werden.',

  // banner
  'banner.detected_source_language': 'Erkannte Ausgangssprache',
  'banner.source_language': 'Ausgangssprache',
}
