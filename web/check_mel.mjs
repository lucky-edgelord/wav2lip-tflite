// Compare src/mel.js with w2l/audio.py. Dump float32 files from Python first:
//   w = load_wav(path); w.tofile("wav.f32"); melspectrogram(w).tofile("mel.f32")
// then: node check_mel.mjs wav.f32 mel.f32
import { readFileSync } from 'fs';
import { melspectrogram, melChunkStarts } from './src/mel.js';
const wav = new Float32Array(readFileSync(process.argv[2]).buffer.slice(0));
const ref = new Float32Array(readFileSync(process.argv[3]).buffer.slice(0));
const t0 = performance.now(); const m = await melspectrogram(wav); const ms = performance.now() - t0;
let maxd = 0, sum = 0; for (let i = 0; i < ref.length; i++) { const d = Math.abs(ref[i] - m.data[i]); maxd = Math.max(maxd, d); sum += d; }
console.log('frames', m.frames, 'ref', ref.length / 80, 'max diff', maxd.toExponential(2), 'mean', (sum / ref.length).toExponential(2), 'ms', ms.toFixed(0));
console.log('chunks', melChunkStarts(m.frames, 25).length);
