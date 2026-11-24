// node test/w4test.mjs <wasm> <w2l.bin> <blaze.bin> <refdir> [threads]
import { Worker } from 'node:worker_threads';
import { readFileSync } from 'node:fs';
import { W4 } from '../src/w4.js';
const [wasm, w2l, blaze, ref, thr = '4'] = process.argv.slice(2);
const f32 = (p) => { const b = readFileSync(p); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };
const eng = await W4.create({ wasm: readFileSync(wasm), threads: +thr, makeWorker: () => new Worker(new URL('./node_worker.mjs', import.meta.url)), log: console.log });
const N = 16, u8 = (p) => { const m = readFileSync(p); return new Uint8Array(m.buffer.slice(m.byteOffset, m.byteOffset + m.length)); };
await eng.loadModel(0, u8(w2l), N); await eng.loadModel(1, u8(blaze), N);
const mel = f32(`${ref}/mel.f32`), face = f32(`${ref}/face.f32`), pred = f32(`${ref}/pred.f32`);
await eng.run(0, [mel, face], N);
let t0 = performance.now(); const [out] = await eng.run(0, [mel, face], N); const ms = performance.now() - t0;
let se = 0; for (let i = 0; i < out.length; i++) se += (out[i] - pred[i]) ** 2;
console.log(`wav2lip: ${(ms / N).toFixed(0)} ms/frame with ${eng.threads} workers, PSNR vs FP32 ${(10 * Math.log10(1 / (se / out.length))).toFixed(2)} dB`);
const [r, s] = await eng.run(1, [f32(`${ref}/bin.f32`)], N); const sc = f32(`${ref}/bsc.f32`);
let ns = 0, cs = 0; for (let i = 0; i < s.length; i++) { cs += (s[i] - sc[i]) ** 2; ns += sc[i] ** 2; }
console.log(`blazeface: score rel.err ${Math.sqrt(cs / ns).toFixed(3)}`);
eng.terminate();
