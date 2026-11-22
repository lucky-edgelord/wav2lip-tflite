// Mel spectrogram matching w2l/audio.py (Wav2Lip's hparams) exactly:
// 16 kHz, pre-emphasis 0.97, STFT n_fft=800 hop=200 periodic Hann, centered with
// zero padding, Slaney mel (80 bands, 55-7600 Hz), dB with ref 20, symmetric
// normalisation to [-4, 4]. Verified against w2l/audio.py with check_mel.mjs.

export const SAMPLE_RATE = 16000;
const N_FFT = 800, HOP = 200, N_MELS = 80, FMIN = 55, FMAX = 7600;
const N_BINS = N_FFT / 2 + 1;
const MEL_STEP = 16;

function hzToMel(f) {
  const fsp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fsp, logstep = Math.log(6.4) / 27;
  return f >= minLogHz ? minLogMel + Math.log(f / minLogHz) / logstep : f / fsp;
}
function melToHz(m) {
  const fsp = 200 / 3, minLogHz = 1000, minLogMel = minLogHz / fsp, logstep = Math.log(6.4) / 27;
  return m >= minLogMel ? minLogHz * Math.exp(logstep * (m - minLogMel)) : fsp * m;
}

function melFilterbank() {
  const fft = Float64Array.from({ length: N_BINS }, (_, k) => (k * SAMPLE_RATE) / N_FFT);
  const mMin = hzToMel(FMIN), mMax = hzToMel(FMAX);
  const pts = Float64Array.from({ length: N_MELS + 2 }, (_, i) => melToHz(mMin + ((mMax - mMin) * i) / (N_MELS + 1)));
  const fb = new Float32Array(N_MELS * N_BINS);
  for (let m = 0; m < N_MELS; m++) {
    const lo = pts[m], c = pts[m + 1], hi = pts[m + 2];
    const enorm = 2 / (hi - lo);
    for (let k = 0; k < N_BINS; k++) {
      const w = Math.max(0, Math.min((fft[k] - lo) / (c - lo), (hi - fft[k]) / (hi - c)));
      fb[m * N_BINS + k] = w * enorm;
    }
  }
  return fb;
}

// Real FFT of size 800 = 32 * 25 via a mixed-radix (2, 5) recursive complex FFT.
function makeFft(n) {
  const cos = new Float64Array(n), sin = new Float64Array(n);
  for (let i = 0; i < n; i++) { cos[i] = Math.cos((2 * Math.PI * i) / n); sin[i] = -Math.sin((2 * Math.PI * i) / n); }
  function rec(re, im, off, stride, len, outRe, outIm, outOff) {
    if (len === 1) { outRe[outOff] = re[off]; outIm[outOff] = im[off]; return; }
    const p = len % 2 === 0 ? 2 : 5;
    const m = len / p;
    for (let r = 0; r < p; r++) rec(re, im, off + r * stride, stride * p, m, outRe, outIm, outOff + r * m);
    const tRe = new Float64Array(p), tIm = new Float64Array(p);
    const step = n / len;
    for (let k = 0; k < m; k++) {
      for (let r = 0; r < p; r++) {
        const tw = (r * k * step) % n;
        const xr = outRe[outOff + r * m + k], xi = outIm[outOff + r * m + k];
        tRe[r] = xr * cos[tw] - xi * sin[tw];
        tIm[r] = xr * sin[tw] + xi * cos[tw];
      }
      for (let q = 0; q < p; q++) {
        let sr = 0, si = 0;
        for (let r = 0; r < p; r++) {
          const tw = (r * q * m * step) % n;
          sr += tRe[r] * cos[tw] - tIm[r] * sin[tw];
          si += tRe[r] * sin[tw] + tIm[r] * cos[tw];
        }
        outRe[outOff + q * m + k] = sr;
        outIm[outOff + q * m + k] = si;
      }
    }
  }
  return (re, im, outRe, outIm) => rec(re, im, 0, 1, n, outRe, outIm, 0);
}

/** wav: Float32Array at 16 kHz mono. Returns {data: Float32Array(80*T) row-major [mel][t], frames: T}. */
export async function melspectrogram(wav, onProgress) {
  const n = wav.length;
  const emph = new Float32Array(n);
  for (let i = 0; i < n; i++) emph[i] = wav[i] - (i ? 0.97 * wav[i - 1] : 0);
  const pad = N_FFT / 2;
  const padded = new Float32Array(n + 2 * pad);
  padded.set(emph, pad);
  const T = 1 + Math.floor((padded.length - N_FFT) / HOP);
  const win = Float64Array.from({ length: N_FFT }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / N_FFT));
  const fb = melFilterbank();
  const fft = makeFft(N_FFT);
  const re = new Float64Array(N_FFT), im = new Float64Array(N_FFT);
  const oRe = new Float64Array(N_FFT), oIm = new Float64Array(N_FFT);
  const mag = new Float64Array(N_BINS);
  const out = new Float32Array(N_MELS * T);
  const minLevel = Math.exp((-100 / 20) * Math.log(10));
  for (let t = 0; t < T; t++) {
    const s = t * HOP;
    for (let i = 0; i < N_FFT; i++) { re[i] = padded[s + i] * win[i]; im[i] = 0; }
    fft(re, im, oRe, oIm);
    for (let k = 0; k < N_BINS; k++) mag[k] = Math.hypot(oRe[k], oIm[k]);
    for (let m = 0; m < N_MELS; m++) {
      let v = 0;
      const row = m * N_BINS;
      for (let k = 0; k < N_BINS; k++) v += fb[row + k] * mag[k];
      const db = 20 * Math.log10(Math.max(minLevel, v)) - 20;
      const norm = 8 * ((db + 100) / 100) - 4;
      out[m * T + t] = Math.min(4, Math.max(-4, norm));
    }
    if (onProgress && t % 500 === 0) { onProgress(t / T); await new Promise((r) => setTimeout(r)); }
  }
  return { data: out, frames: T };
}

/** One (80 x 16) window per output frame, same indexing as Wav2Lip. Returns start columns. */
export function melChunkStarts(frames, fps) {
  const starts = [];
  const mult = 80 / fps;
  for (let i = 0; ; i++) {
    const s = Math.floor(i * mult);
    if (s + MEL_STEP > frames) { starts.push(frames - MEL_STEP); break; }
    starts.push(s);
  }
  return starts;
}

/** Writes the chunk at `start` into dst at offset, as (80, 16, 1) channels-last. */
export function writeMelChunk(mel, start, dst, offset) {
  const T = mel.frames;
  for (let m = 0; m < N_MELS; m++)
    for (let j = 0; j < MEL_STEP; j++) dst[offset + m * MEL_STEP + j] = mel.data[m * T + start + j];
}
