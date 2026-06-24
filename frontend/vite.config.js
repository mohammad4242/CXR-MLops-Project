import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// During local development the React dev server runs on :5173 while FastAPI
// runs on :8000. The browser, however, only ever calls "/api/..." — Vite
// proxies those calls to the backend and strips the "/api" prefix, so the
// exact same fetch code works in production behind Nginx (see nginx.conf).
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
})
