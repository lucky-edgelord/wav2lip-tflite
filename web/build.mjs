// Builds a self-contained static site in dist/: the page, one bundled app.js,
// and the Wasm/model files LiteRT.js and MediaPipe load at runtime (they are
// served from the same origin, so the page needs no CDN at runtime).
import { build } from 'esbuild';
import { cpSync, mkdirSync, rmSync, existsSync, createWriteStream } from 'fs';
import { get } from 'https';

const out = 'dist';
rmSync(out, { recursive: true, force: true });
mkdirSync(`${out}/litert`, { recursive: true });
mkdirSync(`${out}/mediapipe`, { recursive: true });
mkdirSync(`${out}/models`, { recursive: true });
mkdirSync(`${out}/assets`, { recursive: true });

await build({
  entryPoints: ['src/app.js'], bundle: true, format: 'esm', minify: true,
  target: 'es2022', outfile: `${out}/app.js`, legalComments: 'none',
});
cpSync('index.html', `${out}/index.html`);

// LiteRT.js picks the relaxed-SIMD build or the compat build at runtime; ship both.
for (const f of ['litert_wasm_internal', 'litert_wasm_compat_internal'])
  for (const ext of ['.js', '.wasm']) cpSync(`node_modules/@litertjs/core/wasm/${f}${ext}`, `${out}/litert/${f}${ext}`);
for (const ext of ['.js', '.wasm'])
  cpSync(`node_modules/@mediapipe/tasks-vision/wasm/vision_wasm_internal${ext}`, `${out}/mediapipe/vision_wasm_internal${ext}`);

const face = 'models/blaze_face_short_range.tflite';
if (!existsSync(face)) {
  mkdirSync('models', { recursive: true });
  await new Promise((res, rej) => get('https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite',
    (r) => (r.statusCode === 200 ? r.pipe(createWriteStream(face)).on('finish', res) : rej(new Error(`HTTP ${r.statusCode}`)))).on('error', rej));
}
cpSync(face, `${out}/${face}`);
// INT8 Wav2Lip models + quantization manifest (made by scripts/export_web_models.py)
cpSync('assets', `${out}/assets`, { recursive: true });
console.log(`built ${out}/`);
