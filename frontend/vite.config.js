import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { env } from 'node:process'

const apiTarget = 'http://127.0.0.1:' + (env.AGRIVISION_API_PORT || '8000')

export default defineConfig({
  worker: { format: 'es' },
  plugins: [
    react(),
    tailwindcss(),
  ],
  server: {
    host: '127.0.0.1', port: Number(env.AGRIVISION_WEB_PORT || 5173), strictPort: true,
    proxy: {
      '/api': { target: apiTarget, rewrite: (path) => path.replace(/^\/api/, '') },
      '/outputs': apiTarget,
    },
  },
})
