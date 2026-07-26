import { apiGet } from './client'

export interface AppConfig {
  language: 'en' | 'de'
}

export const getConfig = () => apiGet<AppConfig>('/config')
