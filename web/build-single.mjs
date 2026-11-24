// Builds single/w2l_int4.html: ONE file with the UI, JS, both WASM builds, the worker and both INT4 models inside.
import { build } from 'esbuild';
import { readFileSync, writeFileSync, mkdirSync, rmSync } from 'fs';
const out = process.argv[2] || 'single'; rmSync(out, { recursive: true, force: true }); mkdirSync(out, { recursive: true });
const b64 = (p) => readFileSync(p).toString('base64');
const worker = (await build({ entryPoints: ['src/w4worker.js'], bundle: true, format: 'iife', minify: true, write: false, target: 'es2022' })).outputFiles[0].text;
const app = (await build({
  entryPoints: ['src/int4.js'], bundle: true, format: 'iife', minify: true, write: false, target: 'es2022', legalComments: 'none',
  define: { __BUILD__: JSON.stringify(`single file · built ${new Date().toISOString().slice(0, 16).replace('T', ' ')} UTC`), __MANIFEST__: '{}',
    __WASM_ST_B64__: JSON.stringify(b64('assets/w4_st.wasm')), __WORKER__: JSON.stringify(worker) },
})).outputFiles[0].text.replace(/<\/script/gi, '<\\/script');
const emb = [['w2l', 'assets/w2l_w4.bin', 'Wav2Lip INT4'], ['blaze', 'assets/blaze_w4.bin', 'BlazeFace INT4']]
  .map(([k, p, l]) => `<script type="text/plain" id="emb-${k}" data-label="${l}">${b64(p)}</script>`).join('\n');
let html = readFileSync('index.html', 'utf8')
  .replace(/<script>\n\/\/ Enable cross-origin isolation[\s\S]*?<\/script>/, '')   // threads no longer need isolation
  .replace('<script type="module" src="./int4.js"></script>', () => `${emb}\n<script>${app}</script>`)
  .replace('Weights are INT4 (about 20 MB, downloaded once and saved on this device); activations are INT8 inside the engine. Needs a browser with WebAssembly SIMD and threads (Chrome, Edge, Firefox, Safari 16.4+).',
           'Everything, both INT4 models included, lives in this one file. Weights are INT4, activations INT8 inside the engine. 4 worker threads, no special hosting needed.')
  .replace(/<a href="\.\/litert\.html">.*?<\/a>/, 'the LiteRT.js version');
writeFileSync(`${out}/index.html`, html);
console.log(`built ${out}/index.html ${(html.length / 1048576).toFixed(1)} MB`);
