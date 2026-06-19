// Mirrors translator/main.py TranslationRequest/TranslationResponse and the
// /languages payload. Field names match the JSON exactly.

export interface Language {
  code: string
  name: string
}

export interface DetectedLanguage {
  name: string
  flag: string
}

export interface TranslateRequest {
  text: string
  target_lang: string
  source_lang?: string
}

export interface TranslateResponse {
  translation: string
  detected_language: DetectedLanguage
}
