import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import sri from 'vite-plugin-sri-gen'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), sri()],
  base: '/Sparse2Full_Official/', // important for GitHub Pages deployment matching the repository name
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/tests/setup.js',
  },
})