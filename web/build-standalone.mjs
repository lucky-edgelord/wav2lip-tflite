// Builds standalone/: one self-contained index.html (JS, CSS, WASM engine and worker inlined) + the two INT4 model files.
//   node build-standalone.mjs      then serve standalone/ over http(s) (python3 serve.py in that folder works locally)
import { build } from 'esbuild';
import { readFileSync, writeFileSync, mkdirSync, rmSync, cpSync, statSync } from 'fs';
const out = 'standalone';
rmSync(out, { recursive: true, force: true }); mkdirSync(out, { recursive: true });
const worker = (await build({ entryPoints: ['src/w4worker.js'], bundle: true, format: 'iife', minify: true, write: false, target: 'es2022' })).outputFiles[0].text;
const manifest = {};
for (const [k, file, label] of [['w2l', 'w2l_w4.bin', 'Wav2Lip INT4'], ['blaze', 'blaze_w4.bin', 'BlazeFace INT4']]) {
  cpSync(`assets/${file}`, `${out}/${file}`); manifest[k] = { file, bytes: statSync(`assets/${file}`).size, label };
}
const app = (await build({
  entryPoints: ['src/int4.js'], bundle: true, format: 'iife', minify: true, write: false, target: 'es2022', legalComments: 'none',
  define: { __BUILD__: JSON.stringify(`standalone · built ${new Date().toISOString().slice(0, 16).replace('T', ' ')} UTC`), __MANIFEST__: JSON.stringify(manifest),
    __WASM_B64__: JSON.stringify(readFileSync('assets/w4.wasm').toString('base64')), __WORKER__: JSON.stringify(worker) },
})).outputFiles[0].text.replace(/<\/script/gi, '<\\/script');
let html = readFileSync('index.html', 'utf8').replace('<script type="module" src="./int4.js"></script>', () => `<script>${app}</script>`);
writeFileSync(`${out}/index.html`, html);
cpSync('coi-sw.js', `${out}/coi-sw.js`);
writeFileSync(`${out}/serve.py`, `# Local server with the headers WebAssembly threads need:  python3 serve.py  ->  http://localhost:8000
import http.server, socketserver
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin'); self.send_header('Cross-Origin-Embedder-Policy', 'credentialless')
        self.send_header('Cache-Control', 'no-store'); super().end_headers()
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(('', 8000), H) as s: print('http://localhost:8000'); s.serve_forever()
`);
console.log(`built ${out}/ (index.html ${(html.length / 1024).toFixed(0)} KB)`);
