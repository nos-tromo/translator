import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'

const BACKEND = process.env.TRANSLATOR_BACKEND_ORIGIN ?? 'http://localhost:8000'

export default defineConfig({
  plugins: [react()],
  base: '/translator/',
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      // SPA emits /translator/api/... ; backend serves /api/... — strip here in dev
      // (in prod the app's own nginx strips it).
      '/translator/api': {
        target: BACKEND,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/translator/, ''),
      },
    },
  },
  test: {
    environment: 'happy-dom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
  },
})
