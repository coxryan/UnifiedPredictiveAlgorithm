
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// IMPORTANT:
// If your repo name is not 'UnifiedPredictiveAlgorithm', change base accordingly.
export default defineConfig({
  plugins: [react()],
  base: '/UnifiedPredictiveAlgorithm/',
})
