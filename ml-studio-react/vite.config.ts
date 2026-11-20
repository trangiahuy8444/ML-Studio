import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
export default defineConfig({base:'./',plugins:[react(),tailwindcss()],build:{sourcemap:false,rollupOptions:{output:{inlineDynamicImports:true,format:'iife',entryFileNames:'assets/app.js',assetFileNames:'assets/[name][extname]'}}}});
