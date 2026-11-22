// Wav2Lip in the browser: MediaPipe BlazeFace -> 96x96 crop -> Wav2Lip (LiteRT.js,
// WebGPU or CPU) -> paste back -> MP4 via WebCodecs (Mediabunny). Mirrors infer.py.
// Every step logs to the in-page console (src/log.js); open it with the Log button.
import { L, envReport, mountLogUI, memory, openDrawer } from './log.js';
import { loadLiteRt, loadAndCompile, Tensor, getWebGpuDevice, supportsFeature } from '@litertjs/core';
import { FilesetResolver, FaceDetector } from '@mediapipe/tasks-vision';
import {
  Input, ALL_FORMATS, BlobSource, CanvasSink, AudioBufferSink, Output, Mp4OutputFormat, BufferTarget,
  CanvasSource, AudioBufferSource, QUALITY_HIGH, getFirstEncodableVideoCodec, getFirstEncodableAudioCodec,
} from 'mediabunny';
import { melspectrogram, melChunkStarts, writeMelChunk, SAMPLE_RATE } from './mel.js';

/* global __BUILD__ */
const BUILD = typeof __BUILD__ === 'string' ? __BUILD__ : 'dev';
const IMG = 96;
const $ = (id) => document.getElementById(id);
const base = (p) => new URL(p, location.href).href;
const files = { face: null, audio: null, model: null };
let manifest = null; const modelBytes = {}; // 'gpu' | 'cpu' | 'custom' -> Uint8Array
const SIMULATE_GPU_LOSS = new URLSearchParams(location.search).get('simulate') === 'gpu-loss'; // test hook
let cpuThreads = 1;
let running = false, cancelled = false, resultBlob = null, liteRtReady = null, detector = null, detectorTs = 0, fileset = null;

mountLogUI('wav2lip-log');
envReport(BUILD).catch((e) => L.warn('Environment report failed:', e));

// ---------- file pickers (click or drop) ----------
const fmtSize = (n) => (n > 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1e3))} KB`);
async function pickFile(kind, file) {
  L.info(`Picked ${kind}:`, { name: file.name, type: file.type || '(none)', size: file.size, lastModified: new Date(file.lastModified).toISOString() });
  if (kind === 'model') return setFile(kind, file);
  const row = document.querySelector(`.pick[data-kind="${kind}"]`);
  row.querySelector('.name').textContent = `Reading ${file.name}…`;
  try {
    // Copy into memory now: on Android, picked files (e.g. from Photos) can fail to read later.
    const buf = await L.step(`read ${kind} file into memory`, () => file.arrayBuffer());
    if (!buf.byteLength) throw new Error('the file is empty (0 bytes)');
    setFile(kind, new File([buf], file.name, { type: file.type }));
  } catch (e) {
    setFile(kind, null);
    status(`Could not read ${file.name} (${e.message}). If it is only in the cloud (Google Photos/Drive), save it to the phone first.`, true);
  }
}
function setFile(kind, file, note) {
  files[kind] = file;
  const row = document.querySelector(`.pick[data-kind="${kind}"]`);
  row.classList.toggle('has', !!file);
  row.querySelector('.name').textContent = file ? `${file.name} · ${fmtSize(file.size)}${note ? ` · ${note}` : ''}` : row.dataset.empty;
  if (kind === 'face' && file) showPoster(file);
  if (kind === 'model') { delete modelBytes.custom; L.info(file ? 'Using your own model file.' : 'Using the built-in INT8 model.'); }
  $('run').disabled = !(files.face && files.audio) || running;
}
for (const row of document.querySelectorAll('.pick')) {
  const input = row.querySelector('input');
  row.dataset.empty = row.querySelector('.name').textContent;
  input.addEventListener('change', () => input.files[0] && pickFile(row.dataset.kind, input.files[0]));
  row.addEventListener('dragover', (e) => { e.preventDefault(); row.classList.add('over'); });
  row.addEventListener('dragleave', () => row.classList.remove('over'));
  row.addEventListener('drop', (e) => {
    e.preventDefault(); row.classList.remove('over');
    if (e.dataTransfer.files[0]) pickFile(row.dataset.kind, e.dataTransfer.files[0]);
  });
}

// ---------- bundled INT8 models, fetched from this site with progress ----------
const loading = {};
const CACHE = 'wav2lip-models-v1';
async function download(url, entry, label) {
  // Abort if no data arrives for 15 s: a dropped mobile connection can otherwise hang forever.
  const ctl = new AbortController();
  let watchdog;
  const arm = () => { clearTimeout(watchdog); watchdog = setTimeout(() => ctl.abort(new Error('connection stalled: no data for 15 s')), 15000); };
  arm();
  const s = performance.now();
  try {
    const res = await fetch(url, { cache: 'no-store', signal: ctl.signal });
    L.info(`HTTP ${res.status} ${url}`, { contentLength: res.headers.get('content-length'), contentEncoding: res.headers.get('content-encoding') || 'none', type: res.headers.get('content-type') });
    if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
    // The server may gzip the file, so Content-Length can be the compressed size:
    // count decompressed bytes against the size recorded in models.json.
    const reader = res.body.getReader(), chunks = [];
    let got = 0, lastPaint = 0, nextLog = 0.1;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); got += value.length; arm();
      if (got / entry.bytes >= nextLog) { L.debug(`  ${label}: ${(got / 1e6).toFixed(1)} MB (${Math.floor((got / entry.bytes) * 100)}%)`); nextLog += 0.1; }
      if (performance.now() - lastPaint > 100) { lastPaint = performance.now(); dl({ label, url, got, total: entry.bytes }); }
    }
    if (got !== entry.bytes) throw new Error(`incomplete download (${fmtSize(got)} of ${fmtSize(entry.bytes)})`);
    const buf = new Uint8Array(got);
    for (let o = 0, i = 0; i < chunks.length; o += chunks[i].length, i++) buf.set(chunks[i], o);
    const secs = (performance.now() - s) / 1000;
    L.info(`Downloaded ${label}: ${fmtSize(got)} in ${secs.toFixed(1)} s (${((got * 8) / 1e6 / secs).toFixed(1)} Mbit/s)`);
    return buf;
  } catch (e) {
    throw ctl.signal.aborted ? ctl.signal.reason : e;
  } finally { clearTimeout(watchdog); }
}
async function fetchModel(key) {
  if (modelBytes[key]) return modelBytes[key];
  return (loading[key] ??= (async () => {
    try {
      if (!manifest) {
        const r = await fetch(base('./assets/models.json'), { cache: 'no-store' });
        if (!r.ok) throw new Error(`models.json: HTTP ${r.status}`);
        manifest = await r.json();
        L.debug('Model manifest:', manifest);
      }
      const entry = manifest[key], url = base(`./assets/${entry.file}`);
      const label = entry.precision === 'int8' ? 'Full INT8 (CPU)' : 'INT8 weights (GPU)';
      const cacheKey = `${url}?bytes=${entry.bytes}`;
      let buf = null;
      try { // saved copy from an earlier visit
        const hit = await (await caches.open(CACHE)).match(cacheKey);
        if (hit) {
          const b = new Uint8Array(await hit.arrayBuffer());
          if (b.length === entry.bytes) buf = b;
          else L.warn(`Saved model has the wrong size (${b.length}), downloading again.`);
        } else L.info(`No saved copy of ${label}; downloading.`);
      } catch (e) { L.warn('Cache Storage unavailable, the model will not be saved on this device:', e); }
      if (buf) {
        L.info(`Loaded ${label} from this device (${fmtSize(buf.length)}).`);
        dl({ label, url, got: buf.length, total: entry.bytes, done: true, cached: true });
        return (modelBytes[key] = buf);
      }
      dl({ label, url, got: 0, total: entry.bytes });
      for (let attempt = 1; ; attempt++) {
        try { L.info(`Downloading ${label}, attempt ${attempt}/3`); buf = await download(url, entry, label); break; }
        catch (e) {
          L.warn(`Download attempt ${attempt} failed:`, e);
          if (attempt === 3) throw new Error(`${e.message} (after 3 tries)`);
          dl({ label, url, got: 0, total: entry.bytes, note: `connection dropped (${e.message}), retrying ${attempt + 1}/3…` });
          await new Promise((r) => setTimeout(r, 1500 * attempt));
        }
      }
      try { await (await caches.open(CACHE)).put(cacheKey, new Response(buf)); L.info(`Saved ${label} on this device for next time.`); }
      catch (e) { L.warn('Could not save the model on this device (storage full or blocked):', e); }
      modelBytes[key] = buf;
      dl({ label, url, got: buf.length, total: entry.bytes, done: true });
      return buf;
    } catch (e) {
      L.error(`Model download failed (${key}):`, e);
      dl({ error: `Model download failed: ${e.message}` });
      throw new Error(`model download failed: ${e.message}`, { cause: e });
    } finally { delete loading[key]; }
  })());
}
function dl({ label, url, got, total, done, error, cached, note }) {
  const box = $('dl'), pct = total ? Math.min(100, (got / total) * 100) : 0;
  box.classList.toggle('err', !!error); box.classList.toggle('ok', !!done);
  $('retry').hidden = !error;
  if (error) { $('dl-title').textContent = error; $('model-state').textContent = 'Model: error'; return; }
  $('dl-title').textContent = done ? `${label} model ready${cached ? ' (saved on this device)' : ''}` : note ? `${label}: ${note}` : `Downloading ${label} model`;
  $('dl-bar').style.width = `${pct.toFixed(1)}%`;
  $('dl-num').textContent = `${(got / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB · ${Math.floor(pct)}%`;
  $('dl-src').textContent = `from ${url.replace(/^https?:\/\//, '')}`;
  $('model-state').textContent = done ? `${label}: ready` : `${label}: ${Math.floor(pct)}%`;
  $('model-state').classList.toggle('ok', !!done);
}
// If WebGPU crashed on this device with this model, Auto mode starts on CPU next time.
const gpuLostKey = () => `w2l-gpu-lost:${manifest?.gpu?.file || 'gpu'}`;
const gpuLostBefore = () => { try { return !!localStorage.getItem(gpuLostKey()); } catch { return false; } };
const rememberGpuLost = (why) => { try { localStorage.setItem(gpuLostKey(), `${new Date().toISOString()} ${why}`); } catch {} };
const preferredKey = () => ($('accel').value === 'wasm' || !navigator.gpu || ($('accel').value === 'auto' && gpuLostBefore()) ? 'cpu' : 'gpu');
fetchModel(preferredKey()).catch(() => {});

// ---------- preview ----------
const stage = $('stage'), view = $('view'), vctx = view.getContext('2d');
async function showPoster(file) {
  try {
    let bmp;
    if (file.type.startsWith('image/')) bmp = await createImageBitmap(file);
    else {
      const input = new Input({ formats: ALL_FORMATS, source: new BlobSource(file) });
      const track = await input.getPrimaryVideoTrack();
      if (!track) throw new Error('no video track found');
      L.info('Face video:', { container: (await input.getFormat()).name, codec: track.codec, size: `${track.displayWidth}×${track.displayHeight}`, rotation: track.rotation, durationS: +(await track.computeDuration()).toFixed(2) });
      const wc = await new CanvasSink(track).getCanvas(await track.getFirstTimestamp());
      if (!wc) throw new Error('could not decode the first frame');
      bmp = wc.canvas;
    }
    view.width = bmp.width; view.height = bmp.height; vctx.drawImage(bmp, 0, 0);
    stage.classList.add('ready'); $('result').hidden = true; view.hidden = false;
  } catch (e) { L.error('Preview failed:', e); status(`Could not read that face file: ${e.message}`, true); }
}
function status(msg, isError) {
  $('status').textContent = msg;
  $('status').classList.toggle('err', !!isError);
  $('open-log').hidden = !isError;
}
function hud(msg) { $('hud-r').textContent = msg; }
function progress(f) { $('bar').style.width = `${Math.round(f * 100)}%`; }

// ---------- runtime setup ----------
async function getDetector(forceCpu = false) {
  if (detector && !forceCpu) return detector;
  fileset ??= await L.step('load MediaPipe Wasm', () => FilesetResolver.forVisionTasks(base('./mediapipe')));
  for (const delegate of forceCpu ? ['CPU'] : ['GPU', 'CPU']) {
    try {
      const d = await L.step(`create face detector (${delegate})`, () => FaceDetector.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: base('./models/blaze_face_short_range.tflite'), delegate },
        runningMode: 'VIDEO', minDetectionConfidence: 0.5,
      }));
      try { detector?.close(); } catch {}
      detector = d; detector.delegateName = delegate;
      return detector;
    } catch (e) {
      if (delegate === 'CPU') throw e;
      L.warn('GPU face detector failed, falling back to CPU.');
    }
  }
}

async function compile(bytes, accelerator) {
  const opts = accelerator === 'wasm' ? { accelerator, cpuOptions: { numThreads: cpuThreads } } : { accelerator };
  const model = await L.step(`compile Wav2Lip for ${accelerator}${accelerator === 'wasm' ? `, ${cpuThreads} thread(s)` : ''} (${fmtSize(bytes.length)})`, () => loadAndCompile(bytes, opts));
  const ins = model.getInputDetails(), outs = model.getOutputDetails?.() || [];
  L.debug('Model inputs:', ins.map((d) => ({ name: d.name, index: d.index, dtype: d.dtype, shape: [...d.shape] })));
  L.debug('Model outputs:', outs.map((d) => ({ name: d.name, dtype: d.dtype, shape: [...d.shape] })));
  const mel = ins.find((d) => d.shape.length === 4 && d.shape[1] === 80 && d.shape[2] === 16);
  const face = ins.find((d) => d.shape.length === 4 && d.shape[3] === 6);
  if (!mel || !face) throw new Error('This does not look like a channels-last Wav2Lip model from scripts/convert.py.');
  if (accelerator === 'webgpu') {
    try {
      const dev = await getWebGpuDevice?.();
      dev?.lost?.then((info) => L.error('WebGPU device lost:', { reason: info.reason, message: info.message }));
      dev?.addEventListener?.('uncapturederror', (ev) => L.error('WebGPU error:', ev.error?.message || ev.error));
    } catch (e) { L.debug('Could not attach WebGPU device listeners:', e); }
  }
  return { model, accelerator, batch: mel.shape[0], melIndex: mel.index, faceIndex: face.index, int8: mel.dtype === 'int8' };
}

async function ensureLiteRt() {
  liteRtReady ??= (async () => {
    // Multithreaded CPU needs cross-origin isolation (coi-sw.js adds the headers) and relaxed SIMD.
    let threads = false;
    if (self.crossOriginIsolated) {
      try { threads = (await supportsFeature('relaxedSimd')) !== false; } catch { threads = false; }
    }
    const mobile = navigator.userAgentData?.mobile ?? /Android|iPhone|iPad/.test(navigator.userAgent);
    cpuThreads = threads ? Math.max(1, Math.min(mobile ? 4 : 8, navigator.hardwareConcurrency || 1)) : 1;
    L.info(`CPU inference: ${threads ? `${cpuThreads} threads` : 'single thread'}`, { crossOriginIsolated: self.crossOriginIsolated, cores: navigator.hardwareConcurrency });
    if (!self.crossOriginIsolated) L.warn('Page is not cross-origin isolated, so CPU inference is single-threaded. Reload once to let the helper service worker enable threads.');
    try {
      await L.step(`load LiteRT.js Wasm runtime (${threads ? 'threaded' : 'single-threaded'})`, () => loadLiteRt(base('./litert/'), { threads }));
    } catch (e) {
      if (!threads) throw e;
      L.warn('Threaded LiteRT failed to load, using single-threaded:', e);
      cpuThreads = 1;
      await L.step('load LiteRT.js Wasm runtime (single-threaded)', () => loadLiteRt(base('./litert/'), { threads: false }));
    }
  })();
  try { await liteRtReady; } catch (e) { liteRtReady = null; throw e; }
}

async function getModel(want, onlyCpu = false) {
  await ensureLiteRt();
  if (!manifest) { try { await fetchModel(preferredKey()); } catch {} }
  let gpuOk = !onlyCpu && want !== 'wasm' && !!navigator.gpu;
  if (gpuOk && want === 'auto' && gpuLostBefore()) {
    L.warn('WebGPU crashed on this device in an earlier run, so Auto starts on CPU. Choose "GPU · WebGPU" in Settings to try the GPU again.');
    gpuOk = false;
  }
  if (want === 'webgpu') { try { localStorage.removeItem(gpuLostKey()); } catch {} }
  if (files.model) { // user's own model: float I/O only (no quantization params available)
    modelBytes.custom ??= new Uint8Array(await files.model.arrayBuffer());
    for (const acc of gpuOk ? ['webgpu', 'wasm'] : ['wasm']) {
      try {
        const m = await compile(modelBytes.custom, acc);
        if (m.int8) throw new Error('Custom full-INT8 models are not supported; use fp32 or int8w.');
        m.label = 'your model';
        return m;
      } catch (e) {
        if (acc === 'wasm' || e.message.includes('INT8') || e.message.includes('Wav2Lip')) throw e;
        L.warn('Your model failed on WebGPU, trying CPU.');
      }
    }
  }
  const tries = gpuOk ? [['gpu', 'webgpu'], ['cpu', 'wasm']] : [['cpu', 'wasm']];
  let lastErr;
  for (const [key, acc] of tries) {
    try {
      const m = await compile(await fetchModel(key), acc);
      if (m.int8) m.q = manifest[key].io;
      m.label = key === 'gpu' ? 'INT8 weights' : 'full INT8';
      L.info(`Wav2Lip ready: ${m.label} on ${acc}, batch ${m.batch}${m.int8 ? ', int8 I/O' : ''}`);
      return m;
    } catch (e) {
      lastErr = e;
      if (acc === 'webgpu') { L.warn('WebGPU path failed, switching to CPU:', e); status('GPU path unavailable, switching to CPU…'); }
    }
  }
  throw lastErr;
}

function quantize(src, p) {
  const out = new Int8Array(src.length), inv = 1 / p.scale;
  for (let i = 0; i < src.length; i++) out[i] = Math.max(-128, Math.min(127, Math.round(src[i] * inv) + p.zero_point));
  return out;
}

async function decodeAudio(file) {
  let audio;
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    try {
      const s = performance.now();
      audio = await ctx.decodeAudioData(await file.arrayBuffer());
      L.info(`Decoded audio with Web Audio (${(performance.now() - s).toFixed(0)} ms)`);
    } finally { ctx.close().catch(() => {}); }
  } catch (e) {
    L.warn('Web Audio could not decode this file, trying WebCodecs (Mediabunny):', e);
    try { audio = await L.step('decode audio (WebCodecs)', () => decodeAudioWebCodecs(file)); }
    catch (e2) { throw new Error(`could not decode the voice file (${e2.message}). Use MP3, M4A/AAC, WAV, or a video with sound.`, { cause: e2 }); }
  }
  if (!audio || !audio.length) throw new Error('no audio found in the voice file');
  L.info('Audio:', { durationS: +audio.duration.toFixed(2), sampleRate: audio.sampleRate, channels: audio.numberOfChannels });
  if (audio.duration < 0.3) throw new Error(`the audio is only ${audio.duration.toFixed(2)} s long`);
  const off = new OfflineAudioContext(1, Math.ceil(audio.duration * SAMPLE_RATE), SAMPLE_RATE);
  const src = off.createBufferSource(); src.buffer = audio; src.connect(off.destination); src.start();
  const mono16k = (await L.step('resample audio to 16 kHz mono', () => off.startRendering())).getChannelData(0);
  let peak = 0; for (let i = 0; i < mono16k.length; i += 7) peak = Math.max(peak, Math.abs(mono16k[i]));
  L.info(`Audio peak level ${peak.toFixed(3)}`);
  if (peak < 1e-4) L.warn('The voice track is silent; the mouth will not move.');
  return { audio, mono16k };
}
async function decodeAudioWebCodecs(file) {
  const input = new Input({ formats: ALL_FORMATS, source: new BlobSource(file) });
  const track = await input.getPrimaryAudioTrack();
  if (!track) throw new Error('no audio track in the voice file');
  L.info('Audio track:', { codec: track.codec, sampleRate: track.sampleRate, channels: track.numberOfChannels });
  const parts = [];
  for await (const { buffer } of new AudioBufferSink(track).buffers()) parts.push(buffer);
  if (!parts.length) throw new Error('the audio track decoded to nothing');
  const len = parts.reduce((a, b) => a + b.length, 0), ch = parts[0].numberOfChannels;
  const out = new AudioBuffer({ length: len, numberOfChannels: ch, sampleRate: parts[0].sampleRate });
  for (let c = 0; c < ch; c++) { let o = 0; for (const p of parts) { out.copyToChannel(p.getChannelData(Math.min(c, p.numberOfChannels - 1)), c, o); o += p.length; } }
  return out;
}

// ---------- frame source: video looped to the audio length, or a still image ----------
async function openFace(file, maxW) {
  const fit = (w, h) => {
    const s = Math.min(1, maxW / w);
    return [Math.max(2, Math.round((w * s) / 2) * 2), Math.max(2, Math.round((h * s) / 2) * 2)]; // even sizes for H.264
  };
  if (file.type.startsWith('image/')) {
    const bmp = await createImageBitmap(file);
    const [w, h] = fit(bmp.width, bmp.height);
    L.info('Face image:', { size: `${bmp.width}×${bmp.height}`, output: `${w}×${h}` });
    return { w, h, fps: null, frames: async function* (n) { for (let i = 0; i < n; i++) yield bmp; } };
  }
  const input = new Input({ formats: ALL_FORMATS, source: new BlobSource(file) });
  const track = await input.getPrimaryVideoTrack();
  if (!track) throw new Error('No video track in the face file.');
  if (!(await track.canDecode())) throw new Error(`This browser cannot decode ${track.codec} video. Try an H.264 MP4.`);
  const [w, h] = fit(track.displayWidth, track.displayHeight);
  const start = await track.getFirstTimestamp(), end = await track.computeDuration();
  const stats = await track.computePacketStats(120);
  L.info('Face video track:', { codec: track.codec, size: `${track.displayWidth}×${track.displayHeight}`, output: `${w}×${h}`, fps: +stats.averagePacketRate.toFixed(2), startS: start, endS: +end.toFixed(3) });
  const sink = new CanvasSink(track, { width: w, height: h, fit: 'fill' });
  return {
    w, h, fps: stats.averagePacketRate || null,
    frames: async function* (n, fps) {
      const span = Math.max(1e-3, end - start);
      const times = (function* () { for (let i = 0; i < n; i++) yield start + Math.min((i / fps) % span, span - 1e-3); })();
      let last = null, missing = 0;
      for await (const wc of sink.canvasesAtTimestamps(times)) {
        if (wc) last = wc.canvas; else if (++missing <= 5) L.warn('A video frame could not be decoded; repeating the previous one.');
        if (!last) throw new Error('the first video frame could not be decoded');
        yield last;
      }
      if (missing) L.warn(`${missing} frame(s) could not be decoded and were repeated.`);
    },
  };
}

// ---------- face box: largest face, padded, EMA-smoothed (same as w2l/face.py) ----------
function boxer(padBottom, smooth, stats) {
  let prev = null;
  return async (canvas, fps, w, h) => {
    detectorTs += 1000 / fps;
    let dets;
    try { dets = detector.detectForVideo(canvas, Math.round(detectorTs)).detections; }
    catch (e) {
      L.warn('Face detection failed on this frame, restarting the detector on CPU:', e);
      try { await getDetector(true); dets = detector.detectForVideo(canvas, Math.round((detectorTs += 1))).detections; }
      catch (e2) { L.error('Face detector restart failed; frame left unchanged:', e2); dets = []; }
    }
    if (dets.length) {
      stats.found++;
      const bb = dets.reduce((a, d) => (d.boundingBox.width * d.boundingBox.height > a.width * a.height ? d.boundingBox : a), dets[0].boundingBox);
      let box = [bb.originX, bb.originY, bb.originX + bb.width, bb.originY + bb.height * (1 + padBottom)];
      if (prev && smooth > 0) box = box.map((v, i) => smooth * prev[i] + (1 - smooth) * v);
      if (!prev) L.info('First face found:', { x: Math.round(bb.originX), y: Math.round(bb.originY), w: Math.round(bb.width), h: Math.round(bb.height), score: +(dets[0].categories?.[0]?.score ?? 0).toFixed(2) });
      prev = box;
    } else stats.missed++;
    if (!prev) return null;
    const [x1, y1, x2, y2] = [Math.max(0, prev[0] | 0), Math.max(0, prev[1] | 0), Math.min(w, prev[2] | 0), Math.min(h, prev[3] | 0)];
    return x2 - x1 >= 8 && y2 - y1 >= 8 ? [x1, y1, x2, y2] : null;
  };
}

// ---------- Wav2Lip on one batch, with GPU -> CPU fallback ----------
async function infer(state, melBuf, faceBuf) {
  for (;;) {
    const m = state.m, B = m.batch;
    let live = []; // tensors to free whatever happens
    try {
      const inputs = [];
      inputs[m.melIndex] = new Tensor(m.int8 ? quantize(melBuf, m.q.args_0) : melBuf, [B, 80, 16, 1]);
      inputs[m.faceIndex] = new Tensor(m.int8 ? quantize(faceBuf, m.q.args_1) : faceBuf, [B, IMG, IMG, 6]);
      live = [...inputs];
      // moveTo() hands the data to the GPU and invalidates the CPU tensor.
      const moved = m.accelerator === 'webgpu' ? await Promise.all(inputs.map((x) => x.moveTo('webgpu'))) : inputs;
      live = [...moved];
      if (SIMULATE_GPU_LOSS && m.accelerator === 'webgpu' && ++state.gpuCalls >= 2) throw new Error('simulated WebGPU device loss (?simulate=gpu-loss)');
      const outs = await m.model.run(moved);
      const host = await outs[0].moveTo('wasm');
      const raw = host.toTypedArray();
      let pred;
      if (m.int8) { const p = m.q.output_0; pred = Float32Array.from(raw, (v) => (v - p.zero_point) * p.scale); }
      else pred = raw.slice();
      host.delete(); outs.slice(1).forEach((x) => x.delete());
      if (!state.checked) { state.checked = true; let s = 0; for (let i = 0; i < pred.length; i += 97) s += pred[i]; if (!Number.isFinite(s)) throw new Error('model produced NaN/Inf output'); }
      return pred;
    } catch (e) {
      if (state.m.accelerator !== 'webgpu' || state.fellBack) throw e;
      L.error('Wav2Lip failed on WebGPU; switching to CPU and redoing these frames:', e);
      if (!files.model) rememberGpuLost(e.message);
      status('GPU failed, continuing on CPU…');
      state.fellBack = true;
      state.m = await getModel('wasm', true);
      $('acc').textContent = `Wav2Lip ${state.m.label} on CPU (GPU failed) · batch ${state.m.batch} · ${cpuThreads} thread(s)`;
      return null; // model (and maybe batch size) changed: caller re-chunks
    } finally { for (const x of live) { try { x?.delete(); } catch {} } }
  }
}

// Run k prepared items (laid out back to back in melAll/faceAll) through the current model,
// in chunks of its batch size. Survives a mid-run switch to a model with another batch size.
async function runAll(state, melAll, faceAll, k) {
  const out = new Float32Array(k * IMG * IMG * 3);
  for (let i = 0; i < k;) {
    const B = state.m.batch, c = Math.min(B, k - i);
    const melB = new Float32Array(B * 80 * 16), faceB = new Float32Array(B * IMG * IMG * 6);
    melB.set(melAll.subarray(i * 80 * 16, (i + c) * 80 * 16));
    faceB.set(faceAll.subarray(i * IMG * IMG * 6, (i + c) * IMG * IMG * 6));
    const pred = await infer(state, melB, faceB);
    if (!pred) continue; // model changed; redo this chunk with the new batch size
    out.set(pred.subarray(0, c * IMG * IMG * 3), i * IMG * IMG * 3);
    i += c;
  }
  return out;
}

// ---------- main ----------
async function run() {
  running = true; cancelled = false; resultBlob = null;
  $('run').disabled = true; $('cancel').hidden = false; $('save').hidden = true; $('result').hidden = true; view.hidden = false;
  stage.classList.add('ready', 'busy'); progress(0); hud('SYNCING'); status('Starting…');
  const t = { detect: 0, model: 0, other: 0 };
  let step = 'starting', wake = null;
  L.info('━━ Run started', { face: files.face?.name, voice: files.audio?.name, accel: $('accel').value, fps: $('fps').value || 'auto', pad: $('pad').value, smooth: $('smooth').value, maxWidth: $('maxw').value });
  try {
    try { wake = await navigator.wakeLock?.request('screen'); L.debug('Screen wake lock on.'); } catch (e) { L.debug('No wake lock:', e.message); }
    step = 'loading the face detector'; status('Loading face detector…');
    await getDetector();
    step = 'loading Wav2Lip'; status('Loading Wav2Lip model…');
    const state = { m: await getModel($('accel').value), gpuCalls: 0 };
    step = 'decoding the audio'; status('Decoding audio…');
    const { audio, mono16k } = await decodeAudio(files.audio);
    step = 'computing audio features';
    const mel = await L.step('mel spectrogram', () => melspectrogram(mono16k, (f) => status(`Computing mel spectrogram… ${Math.round(f * 100)}%`)));
    step = 'opening the face video';
    const src = await openFace(files.face, +$('maxw').value);
    const fps = +$('fps').value || (src.fps ? Math.round(src.fps * 100) / 100 : 25);
    const starts = melChunkStarts(mel.frames, fps);
    const n = starts.length, W = 8; // frames prepared per round; the model runs them in its own batch size
    L.info(`Plan: ${n} frames at ${fps} fps, rounds of ${W}, model batch ${state.m.batch}, output ${src.w}×${src.h}`);
    $('acc').textContent = `Wav2Lip ${state.m.label} on ${state.m.accelerator === 'webgpu' ? 'WebGPU' : `CPU, ${cpuThreads} thread(s)`} · batch ${state.m.batch} · face detector on ${detector.delegateName}`;

    view.width = src.w; view.height = src.h;
    hud(`${src.w}×${src.h} · ${fps} FPS`);
    step = 'setting up the video encoder';
    const vcodec = await getFirstEncodableVideoCodec(['avc', 'vp9', 'av1'], { width: src.w, height: src.h });
    const acodec = await getFirstEncodableAudioCodec(['aac', 'opus'], { numberOfChannels: audio.numberOfChannels, sampleRate: audio.sampleRate });
    L.info('Encoders:', { video: vcodec || 'none', audio: acodec || 'none' });
    if (!vcodec) throw new Error(`This browser cannot encode ${src.w}×${src.h} video (WebCodecs). Try Chrome/Edge, or lower "Max output width".`);
    if (!acodec) L.warn('No audio encoder available; the MP4 will be silent.');
    const output = new Output({ format: new Mp4OutputFormat({ fastStart: 'in-memory' }), target: new BufferTarget() });
    const vsrc = new CanvasSource(view, { codec: vcodec, bitrate: QUALITY_HIGH });
    output.addVideoTrack(vsrc, { frameRate: fps });
    let asrc = null;
    if (acodec) { asrc = new AudioBufferSource({ codec: acodec, bitrate: QUALITY_HIGH }); output.addAudioTrack(asrc); }
    await output.start();

    const faceStats = { found: 0, missed: 0 };
    const box = boxer(+$('pad').value, +$('smooth').value, faceStats);
    const pool = Array.from({ length: W }, () => { const c = document.createElement('canvas'); c.width = src.w; c.height = src.h; return c; });
    const crop = new OffscreenCanvas(IMG, IMG), cctx = crop.getContext('2d', { willReadFrequently: true });
    const patch = new OffscreenCanvas(IMG, IMG), pctx = patch.getContext('2d');
    const melBuf = new Float32Array(W * 80 * 16), faceBuf = new Float32Array(W * IMG * IMG * 6);
    const frames = src.frames(n, fps);
    step = 'rendering frames';
    const t0 = performance.now();

    for (let i = 0; i < n && !cancelled; i += W) {
      const count = Math.min(W, n - i), boxes = [], bt = { decode: 0, detect: 0, prep: 0, model: 0, paste: 0, encode: 0 };
      let k = 0;
      melBuf.fill(0); faceBuf.fill(0);
      for (let j = 0; j < count; j++) {
        let s = performance.now();
        const frame = (await frames.next()).value;
        const fc = pool[j], fctx = fc.getContext('2d');
        fctx.drawImage(frame, 0, 0, src.w, src.h);
        bt.decode += performance.now() - s; s = performance.now();
        const b = await box(fc, fps, src.w, src.h);
        bt.detect += performance.now() - s; s = performance.now();
        boxes.push(b);
        if (!b) continue;
        cctx.drawImage(fc, b[0], b[1], b[2] - b[0], b[3] - b[1], 0, 0, IMG, IMG);
        const px = cctx.getImageData(0, 0, IMG, IMG).data, o0 = k * IMG * IMG * 6;
        for (let p = 0, y = 0; y < IMG; y++) for (let x = 0; x < IMG; x++, p += 4) {
          const o = o0 + (y * IMG + x) * 6, bl = px[p + 2] / 255, g = px[p + 1] / 255, r = px[p] / 255;
          faceBuf[o + 3] = bl; faceBuf[o + 4] = g; faceBuf[o + 5] = r; // reference, BGR like OpenCV
          if (y < IMG / 2) { faceBuf[o] = bl; faceBuf[o + 1] = g; faceBuf[o + 2] = r; } // lower half masked
        }
        writeMelChunk(mel, starts[i + j], melBuf, k * 80 * 16);
        k++;
        bt.prep += performance.now() - s;
      }

      if (k) {
        let s = performance.now();
        step = `running Wav2Lip on frames ${i + 1}-${i + count}`;
        const pred = await runAll(state, melBuf, faceBuf, k);
        step = 'rendering frames';
        bt.model = performance.now() - s; s = performance.now();
        const img = pctx.createImageData(IMG, IMG);
        for (let j = 0, q = 0; j < count; j++) {
          const b = boxes[j];
          if (!b) continue;
          const o0 = q++ * IMG * IMG * 3;
          for (let p = 0, e = 0; p < IMG * IMG; p++, e += 3) {
            img.data[p * 4] = pred[o0 + e + 2] * 255; img.data[p * 4 + 1] = pred[o0 + e + 1] * 255;
            img.data[p * 4 + 2] = pred[o0 + e] * 255; img.data[p * 4 + 3] = 255;
          }
          pctx.putImageData(img, 0, 0);
          const fctx = pool[j].getContext('2d');
          fctx.imageSmoothingQuality = 'high';
          fctx.drawImage(patch, b[0], b[1], b[2] - b[0], b[3] - b[1]);
        }
        bt.paste = performance.now() - s;
      }

      const s3 = performance.now();
      step = 'encoding video';
      for (let j = 0; j < count; j++) {
        vctx.drawImage(pool[j], 0, 0);
        await vsrc.add((i + j) / fps, 1 / fps);
      }
      step = 'rendering frames';
      bt.encode = performance.now() - s3;
      t.detect += bt.detect; t.model += bt.model; t.other += bt.decode + bt.prep + bt.paste + bt.encode;
      const done = i + count, el = (performance.now() - t0) / 1000;
      L.debug(`round ${i / W + 1}/${Math.ceil(n / W)} [${state.m.accelerator}] frames ${i + 1}-${done}: faces ${k}/${count} · ms decode ${bt.decode.toFixed(0)} detect ${bt.detect.toFixed(0)} prep ${bt.prep.toFixed(0)} model ${bt.model.toFixed(0)} paste ${bt.paste.toFixed(0)} encode ${bt.encode.toFixed(0)} ${memory()}`);
      progress(done / n);
      status(`Frame ${done} of ${n}`);
      $('fps-out').textContent = (done / el).toFixed(1);
      $('ms-detect').textContent = (t.detect / done).toFixed(1);
      $('ms-model').textContent = (t.model / done).toFixed(1);
      $('ms-other').textContent = (t.other / done).toFixed(1);
    }
    L.info('Faces:', { framesWithFace: faceStats.found, framesWithout: faceStats.missed });
    if (!faceStats.found) L.warn('No face was found in any frame, so nothing was lip-synced. Use a clearer, front-facing shot.');

    if (cancelled) { await output.cancel(); L.info('Run cancelled by user.'); status('Cancelled.'); return; }
    step = 'writing the MP4'; status('Finishing MP4…');
    if (asrc) await L.step('encode audio', () => asrc.add(audio));
    await L.step('finalize MP4', () => output.finalize());
    resultBlob = new Blob([output.target.buffer], { type: 'video/mp4' });
    const v = $('result');
    v.onerror = () => L.error('This browser cannot play the result video back:', v.error?.message || v.error?.code);
    v.src = URL.createObjectURL(resultBlob); v.hidden = false; view.hidden = true;
    $('save').hidden = false;
    const secs = (performance.now() - t0) / 1000;
    L.info(`━━ Done: ${n} frames in ${secs.toFixed(1)} s (${(n / secs).toFixed(1)} fps), MP4 ${fmtSize(resultBlob.size)}`, { avgMs: { detect: +(t.detect / n).toFixed(1), wav2lip: +(t.model / n).toFixed(1), other: +(t.other / n).toFixed(1) } });
    status(`Done: ${n} frames in ${secs.toFixed(1)} s (${(n / secs).toFixed(1)} fps)${acodec ? '' : ' · no audio encoder, video is silent'}.`);
  } catch (e) {
    L.error(`Run failed while ${step}:`, e);
    status(`Error while ${step}: ${e.message || e}`, true);
  } finally {
    running = false; stage.classList.remove('busy');
    try { await wake?.release(); } catch {}
    if (resultBlob) hud('SYNCED'); else if (!cancelled) hud('FAILED');
    $('cancel').hidden = true; $('run').disabled = !(files.face && files.audio);
  }
}

// ---------- save ----------
async function save() {
  if (!resultBlob) return;
  const name = `lipsync-${files.face.name.replace(/\.[^.]+$/, '')}.mp4`;
  const dlc = window.claude ? await window.claude.use('downloads').catch(() => null) : null;
  if (dlc) {
    try { await dlc.save({ filename: name, data: resultBlob }); status('Saved.'); }
    catch (e) { status(e.code === 'declined' ? 'Save cancelled.' : `Could not save (${e.code || e.message}). Long-press the video to save it.`, e.code !== 'declined'); }
    return;
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(resultBlob); a.download = name; document.body.append(a); a.click(); a.remove();
  L.info(`Saved ${name}`);
}

$('run').addEventListener('click', () => run().catch((e) => L.error('Unexpected failure:', e)));
$('cancel').addEventListener('click', () => { cancelled = true; status('Cancelling…'); });
$('save').addEventListener('click', save);
$('open-log').addEventListener('click', openDrawer);
$('webgpu-note').textContent = navigator.gpu ? 'WebGPU online' : 'No WebGPU · CPU mode';
$('webgpu-note').classList.toggle('ok', !!navigator.gpu);
$('accel').addEventListener('change', () => { L.info(`Accelerator set to ${$('accel').value}`); fetchModel(preferredKey()).catch(() => {}); });
$('retry').addEventListener('click', () => fetchModel(preferredKey()).catch(() => {}));
$('clear-model').addEventListener('click', (e) => { e.preventDefault(); setFile('model', null); });
