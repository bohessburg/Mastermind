import react from '@vitejs/plugin-react';
import { fileURLToPath, URL } from 'node:url';
import { defineConfig } from 'vite';
export default defineConfig({
    plugins: [react()],
    server: {
        port: 5173,
        fs: {
            allow: [
                fileURLToPath(new URL('..', import.meta.url)),
            ],
        },
        proxy: {
            '/api': 'http://localhost:8000',
            '/ws': {
                target: 'ws://localhost:8000',
                ws: true,
            },
        },
    },
});
