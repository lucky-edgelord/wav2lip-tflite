// Wav2Lip + BlazeFace in the browser on a hand-written C engine (w4/w4.c) compiled to WebAssembly SIMD.
// Both networks use INT4 weights (groups of 32, packed two per byte); 4 Web Workers share one memory and each
// runs whole frames, so there are no locks inside a network. Pipeline: BlazeFace -> 96x96 crop -> Wav2Lip -> paste -> MP4.
import { L, envReport, mountLogUI, memory, openDrawer } from './log.js';
import {
  Input, ALL_FORMATS, BlobSource, CanvasSink, AudioBufferSink, Output, Mp4OutputFormat, BufferTarget,
  CanvasSource, AudioBufferSource, QUALITY_HIGH, getFirstEncodableVideoCodec, getFirstEncodableAudioCodec,
} from 'mediabunny';
import { melspectrogram, melChunkStarts, writeMelChunk, SAMPLE_RATE } from './mel.js';
import { W4 } from './w4.js';
import { W4GPU } from './w4gpu.js';
import { W4GL } from './w4gl.js';

/* global __BUILD__, __MANIFEST__, __WASM_ST_B64__, __WORKER__ */
const BUILD = typeof __BUILD__ === 'string' ? __BUILD__ : 'dev';
const IMG = 96, THREADS = 4, W = 16; // W = frames per round (= engine batch)
const $ = (id) => document.getElementById(id);
const base = (p) => new URL(p, location.href).href;
const files = { face: null, audio: null };
let manifest = null; const modelBytes = {};
let running = false, cancelled = false, resultBlob = null, engine = null;

mountLogUI('wav2lip-int4-log');
envReport(BUILD).catch((e) => L.warn('Environment report failed:', e));

// ---------- file pickers (click or drop) ----------
const fmtSize = (n) => (n > 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1e3))} KB`);
async function pickFile(kind, file) {
  L.info(`Picked ${kind}:`, { name: file.name, type: file.type || '(none)', size: file.size, lastModified: new Date(file.lastModified).toISOString() });
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
  $('run').disabled = !(files.face && files.audio) || running;
}
for (const row of document.querySelectorAll('.pick')) {
  const input = row.querySelector('input');
  row.dataset.empty = row.querySelector('.name').textContent;
  input.addEventListener('change', () => input.files[0] && pickFile(row.dataset.kind, input.files[0]));
  if (window.webxdc?.importFiles) row.addEventListener('click', async (e) => {   // inside a webxdc client
    e.preventDefault();
    try {
      const fs = await window.webxdc.importFiles({ mimeTypes: row.dataset.kind === 'audio' ? ['audio/*', 'video/*'] : ['video/*', 'image/*'] });
      if (fs && fs[0]) pickFile(row.dataset.kind, fs[0]);
    } catch (err) { L.warn('Could not import a file through webxdc:', err); }
  });
  row.addEventListener('dragover', (e) => { e.preventDefault(); row.classList.add('over'); });
  row.addEventListener('dragleave', () => row.classList.remove('over'));
  row.addEventListener('drop', (e) => {
    e.preventDefault(); row.classList.remove('over');
    if (e.dataTransfer.files[0]) pickFile(row.dataset.kind, e.dataTransfer.files[0]);
  });
}


// ---------- bundled INT8 models, fetched from this site with progress ----------
const loading = {};
const CACHE = 'wav2lip-w4-v1';
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
  const emb = document.getElementById(`emb-${key}`);
  if (emb) {   // single-file build: the model travels inside this page
    const bin = atob(emb.textContent.trim()), buf = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
    const label = emb.dataset.label; dl({ label, url: 'embedded in this page', got: buf.length, total: buf.length, done: true, cached: false });
    $('dl-title').textContent = `${label} model ready (built into this page)`;
    return (modelBytes[key] = buf);
  }
  return (loading[key] ??= (async () => {
    try {
      if (!manifest) {
        if (typeof __MANIFEST__ !== 'undefined') manifest = __MANIFEST__;
        else {
          const r = await fetch(base('./assets/w4.json'), { cache: 'no-store' });
          if (!r.ok) throw new Error(`w4.json: HTTP ${r.status}`);
          manifest = await r.json();
        }
        L.debug('Model manifest:', manifest);
      }
      const entry = manifest[key], url = base(typeof __MANIFEST__ !== 'undefined' ? `./${entry.file}` : `./assets/${entry.file}`);
      const label = entry.label;
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

fetchModel('blaze').then(() => fetchModel('w2l')).catch(() => {});

// ---------- INT4 engine: WebAssembly + workers on shared memory ----------
async function ensureEngine() {
  if (engine) return engine;
  const [w2l, blaze] = [await fetchModel('w2l'), await fetchModel('blaze')];
  const force = new URLSearchParams(location.search).get('backend');   // ?backend=gpu | cpu
  const loadInto = async (eng, tag) => {
    await L.step(`load Wav2Lip INT4 (${tag})`, () => eng.loadModel(0, w2l, W));
    await L.step(`load BlazeFace INT4 (${tag})`, () => eng.loadModel(1, blaze, W));
  };
  // Preference order: WebGPU -> WebGL2 -> WASM threads. Each GPU path is only kept if it survives a real run.
  const z = [new Float32Array(4 * 1280), new Float32Array(4 * 221184)];
  const bench = async (e) => { await e.run(0, z, 1); const t = performance.now(); await e.run(0, z, 4); return (performance.now() - t) / 4; };
  const gpus = [];
  for (const [name, mk, ok] of [['WebGPU', () => W4GPU.create({ log: (m) => L.info(m) }), !!navigator.gpu && force !== 'webgl'],
                                 ['WebGL2', () => W4GL.create({ log: (m) => L.info(m) }), force !== 'gpu' && true]]) {
    if (force === 'cpu' || !ok) continue;
    if (force === 'gpu' && gpus.length) break;
    try {
      const e = await L.step(`start ${name} engine (INT4 dequantized in the shader)`, mk);
      await loadInto(e, name); e.ms = force ? 0 : await bench(e); e.tag = name; gpus.push(e);
      L.info(`${name}: ${e.ms.toFixed(0)} ms/frame`);
      if (force === 'webgl' || force === 'gpu') break;
      if (name === 'WebGPU' && e.ms < 400) break;   // a working, reasonably fast WebGPU: no need to also try WebGL
    } catch (err) { L.warn(`${name} unavailable:`, err); }
  }
  let gpu = gpus.length ? gpus.reduce((x, y) => (y.ms < x.ms ? y : x)) : null, cpu = null;
  if (!gpu || (force !== 'gpu' && force !== 'webgl')) {
    const b64 = typeof __WASM_ST_B64__ !== 'undefined' ? __WASM_ST_B64__ : null;
    const wasm = b64 ? Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)).buffer
      : await L.step('fetch engine (w4_st.wasm)', async () => (await fetch(base('./assets/w4_st.wasm'))).arrayBuffer());
    // Each worker owns a full copy of the engine and models and gets whole frames by message: no shared memory, so no
    // cross-origin isolation is needed and 4 threads work on any host (and when opened from a file).
    const workerUrl = typeof __WORKER__ !== 'undefined' ? URL.createObjectURL(new Blob([__WORKER__], { type: 'text/javascript' })) : base('./w4worker.js');
    cpu = await L.step(`start the WASM engine (${THREADS} threads)`, () => W4.create({
      wasm, threads: THREADS, memMB: 96, log: (m) => L.info(m),
      makeWorker: () => new Worker(workerUrl, typeof __WORKER__ !== 'undefined' ? {} : { type: 'module' }),
    }));
    await loadInto(cpu, 'CPU'); cpu.backend = 'wasm';
  }
  let pick = gpu || cpu;
  if (gpu && cpu && !force) {
    const tc = await bench(cpu);
    L.info(`backend benchmark: ${gpu.tag} ${gpu.ms.toFixed(0)} ms/frame, WASM x${cpu.threads} ${tc.toFixed(0)} ms/frame`);
    pick = gpu.ms < tc ? gpu : cpu;
  }
  for (const e of [...gpus, cpu]) if (e && e !== pick) { try { e.workers?.forEach((w) => w.terminate()); e.device?.destroy(); e.gl?.getExtension('WEBGL_lose_context')?.loseContext(); } catch {} }
  L.info(`INT4 engine ready: ${pick.backend === 'wasm' ? `WASM SIMD x${pick.threads}` : `${pick.tag} (${pick.name})`}`);
  const cn = $('cpu-note'); cn.textContent = pick.backend === 'wasm' ? `WASM SIMD · ${pick.threads} threads` : `${pick.tag} compute`; cn.classList.add('ok');
  return (engine = pick);
}

// ---------- BlazeFace post-processing (anchors, decode, weighted NMS) ----------
const ANCH = (() => {
  const a = [];
  for (const [fm, n] of [[16, 2], [8, 6]]) for (let y = 0; y < fm; y++) for (let x = 0; x < fm; x++) for (let k = 0; k < n; k++) a.push((x + 0.5) / fm, (y + 0.5) / fm);
  return Float32Array.from(a);
})();
function decodeFace(reg, ro, sc, so) { // -> {box:[x1,y1,x2,y2] normalized, score} or null
  const cand = [];
  for (let i = 0; i < 896; i++) {
    const s = 1 / (1 + Math.exp(-Math.max(-50, Math.min(50, sc[so + i]))));
    if (s < 0.5) continue;
    const r = ro + i * 16, cx = reg[r] / 128 + ANCH[2 * i], cy = reg[r + 1] / 128 + ANCH[2 * i + 1], w = reg[r + 2] / 128, h = reg[r + 3] / 128;
    cand.push({ s, b: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] });
  }
  if (!cand.length) return null;
  const best = cand.reduce((a, c) => (c.s > a.s ? c : a)), B = best.b, areaB = (B[2] - B[0]) * (B[3] - B[1]);
  const acc = [0, 0, 0, 0]; let tot = 0;
  for (const c of cand) {
    const iw = Math.max(0, Math.min(c.b[2], B[2]) - Math.max(c.b[0], B[0])), ih = Math.max(0, Math.min(c.b[3], B[3]) - Math.max(c.b[1], B[1]));
    const inter = iw * ih, iou = inter / ((c.b[2] - c.b[0]) * (c.b[3] - c.b[1]) + areaB - inter + 1e-9);
    if (iou > 0.3) { for (let k = 0; k < 4; k++) acc[k] += c.b[k] * c.s; tot += c.s; }
  }
  return { box: acc.map((v) => v / tot), score: best.s };
}

// ---------- face box: best face, padded below the chin, EMA-smoothed ----------
function boxer(padBottom, smooth, stats) {
  let prev = null;
  return (det, w, h) => {
    if (det) {
      stats.found++;
      const [x1, y1, x2, y2] = det.box;
      const box = [x1 * w, y1 * h, x2 * w, (y2 + (y2 - y1) * padBottom) * h];
      if (!prev) L.info('First face found:', { x: Math.round(box[0]), y: Math.round(box[1]), w: Math.round(box[2] - box[0]), h: Math.round(box[3] - box[1]), score: +det.score.toFixed(2) });
      prev = prev && smooth > 0 ? box.map((v, i) => smooth * prev[i] + (1 - smooth) * v) : box;
    } else stats.missed++;
    if (!prev) return null;
    const b = [Math.max(0, prev[0] | 0), Math.max(0, prev[1] | 0), Math.min(w, prev[2] | 0), Math.min(h, prev[3] | 0)];
    return b[2] - b[0] >= 8 && b[3] - b[1] >= 8 ? b : null;
  };
}

// Alpha mask that blends only the lower face; the upper half stays the original (a hard box edge shows as a seam).
const MASK = (() => {
  const m = new Float32Array(IMG * IMG);
  for (let y = 0; y < IMG; y++) for (let x = 0; x < IMG; x++) {
    let v = y < 40 ? 0 : y < 54 ? (y - 40) / 13 : 1;
    const e = Math.min(x, IMG - 1 - x); if (e < 7) v *= (e + 1) / 8;
    if (y >= 88) v *= 1 - 0.6 * ((y - 88) / 7);
    m[y * IMG + x] = v;
  }
  return m;
})();


// ---------- main ----------
async function run() {
  running = true; cancelled = false; resultBlob = null;
  $('run').disabled = true; $('cancel').hidden = false; $('save').hidden = true; $('result').hidden = true; view.hidden = false;
  stage.classList.add('ready', 'busy'); progress(0); hud('SYNCING'); status('Starting…');
  const t = { detect: 0, model: 0, other: 0 };
  let step = 'starting', wake = null;
  L.info('━━ Run started', { face: files.face?.name, voice: files.audio?.name, fps: $('fps').value || 'auto', pad: $('pad').value, smooth: $('smooth').value, maxWidth: $('maxw').value });
  try {
    try { wake = await navigator.wakeLock?.request('screen'); } catch (e) { L.debug('No wake lock:', e.message); }
    step = 'loading the INT4 engine'; status('Loading models and engine…');
    const eng = await ensureEngine();
    step = 'decoding the audio'; status('Decoding audio…');
    const { audio, mono16k } = await decodeAudio(files.audio);
    step = 'computing audio features';
    const mel = await L.step('mel spectrogram', () => melspectrogram(mono16k, (f) => status(`Computing mel spectrogram… ${Math.round(f * 100)}%`)));
    step = 'opening the face video';
    const src = await openFace(files.face, +$('maxw').value);
    const fps = +$('fps').value || (src.fps ? Math.round(src.fps * 100) / 100 : 25);
    const starts = melChunkStarts(mel.frames, fps);
    const n = starts.length;
    L.info(`Plan: ${n} frames at ${fps} fps, rounds of ${W}, output ${src.w}×${src.h}`);
    $('acc').textContent = `Wav2Lip + BlazeFace INT4 · ${engine && engine.backend !== 'wasm' ? `${engine.tag} compute shaders` : `C engine on WebAssembly SIMD · ${engine ? engine.threads : THREADS} thread(s)`}`;

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
    const small = new OffscreenCanvas(128, 128), sctx = small.getContext('2d', { willReadFrequently: true });
    const crop = new OffscreenCanvas(IMG, IMG), cctx = crop.getContext('2d', { willReadFrequently: true });
    const patch = new OffscreenCanvas(IMG, IMG), pctx = patch.getContext('2d');
    const detIn = new Float32Array(W * 128 * 128 * 3), melBuf = new Float32Array(W * 80 * 16), faceBuf = new Float32Array(W * IMG * IMG * 6);
    const frames = src.frames(n, fps);
    step = 'rendering frames';
    const t0 = performance.now();

    for (let i = 0; i < n && !cancelled; i += W) {
      const count = Math.min(W, n - i), boxes = [], bt = { decode: 0, detect: 0, prep: 0, model: 0, paste: 0, encode: 0 };
      let s = performance.now();
      for (let j = 0; j < count; j++) {   // decode + downscale each frame for BlazeFace (128x128, RGB in [-1, 1])
        const frame = (await frames.next()).value;
        pool[j].getContext('2d').drawImage(frame, 0, 0, src.w, src.h);
        sctx.drawImage(pool[j], 0, 0, 128, 128);
        const px = sctx.getImageData(0, 0, 128, 128).data, o = j * 128 * 128 * 3;
        for (let p = 0, q = 0; p < px.length; p += 4, q += 3) { detIn[o + q] = (px[p] - 127.5) / 127.5; detIn[o + q + 1] = (px[p + 1] - 127.5) / 127.5; detIn[o + q + 2] = (px[p + 2] - 127.5) / 127.5; }
      }
      bt.decode = performance.now() - s; s = performance.now();
      step = `finding faces in frames ${i + 1}-${i + count}`;
      const [reg, sc] = await eng.run(1, [detIn], count);
      for (let j = 0; j < count; j++) boxes.push(box(decodeFace(reg, j * 896 * 16, sc, j * 896), src.w, src.h));
      bt.detect = performance.now() - s; s = performance.now();

      melBuf.fill(0); faceBuf.fill(0);
      let k = 0;
      for (let j = 0; j < count; j++) {
        const b = boxes[j]; if (!b) continue;
        cctx.drawImage(pool[j], b[0], b[1], b[2] - b[0], b[3] - b[1], 0, 0, IMG, IMG);
        const px = cctx.getImageData(0, 0, IMG, IMG).data, o0 = k * IMG * IMG * 6;
        for (let p = 0, y = 0; y < IMG; y++) for (let x = 0; x < IMG; x++, p += 4) {
          const o = o0 + (y * IMG + x) * 6, bl = px[p + 2] / 255, g = px[p + 1] / 255, r = px[p] / 255;
          faceBuf[o + 3] = bl; faceBuf[o + 4] = g; faceBuf[o + 5] = r;        // reference, BGR like OpenCV
          if (y < IMG / 2) { faceBuf[o] = bl; faceBuf[o + 1] = g; faceBuf[o + 2] = r; } // lower half masked
        }
        writeMelChunk(mel, starts[i + j], melBuf, k * 80 * 16);
        k++;
      }
      bt.prep = performance.now() - s; s = performance.now();
      if (k) {
        step = `running Wav2Lip on frames ${i + 1}-${i + count}`;
        const [pred] = await eng.run(0, [melBuf, faceBuf], k);
        if (!i) { let sum = 0; for (let q = 0; q < pred.length; q += 97) sum += pred[q]; if (!Number.isFinite(sum)) throw new Error('the engine produced NaN/Inf output'); }
        bt.model = performance.now() - s; s = performance.now();
        const img = pctx.createImageData(IMG, IMG);
        for (let j = 0, q = 0; j < count; j++) {
          const b = boxes[j]; if (!b) continue;
          const o0 = q++ * IMG * IMG * 3;
          for (let p = 0, e = 0; p < IMG * IMG; p++, e += 3) {
            img.data[p * 4] = pred[o0 + e + 2] * 255; img.data[p * 4 + 1] = pred[o0 + e + 1] * 255;
            img.data[p * 4 + 2] = pred[o0 + e] * 255; img.data[p * 4 + 3] = MASK[p] * 255;
          }
          pctx.putImageData(img, 0, 0);
          const fctx = pool[j].getContext('2d'); fctx.imageSmoothingQuality = 'high';
          fctx.drawImage(patch, b[0], b[1], b[2] - b[0], b[3] - b[1]);
        }
        bt.paste = performance.now() - s;
      }
      step = 'encoding video'; const s3 = performance.now();
      for (let j = 0; j < count; j++) { vctx.drawImage(pool[j], 0, 0); await vsrc.add((i + j) / fps, 1 / fps); }
      step = 'rendering frames'; bt.encode = performance.now() - s3;
      t.detect += bt.detect; t.model += bt.model; t.other += bt.decode + bt.prep + bt.paste + bt.encode;
      const done = i + count, el = (performance.now() - t0) / 1000;
      L.debug(`round ${i / W + 1}/${Math.ceil(n / W)} frames ${i + 1}-${done}: faces ${k}/${count} · ms decode ${bt.decode.toFixed(0)} detect ${bt.detect.toFixed(0)} prep ${bt.prep.toFixed(0)} model ${bt.model.toFixed(0)} paste ${bt.paste.toFixed(0)} encode ${bt.encode.toFixed(0)} ${memory()}`);
      progress(done / n); status(`Frame ${done} of ${n}`);
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
  const name = `lipsync-int4-${files.face.name.replace(/\.[^.]+$/, '')}.mp4`;
  if (window.webxdc?.sendToChat) {   // inside a webxdc client: hand the MP4 to the chat
    try { await window.webxdc.sendToChat({ file: { name, blob: resultBlob }, text: 'Lip-synced on-device with Wav2Lip INT4' }); status('Sent to chat.'); }
    catch (e) { L.error('sendToChat failed:', e); status(`Could not send to chat: ${e.message || e}`, true); }
    return;
  }
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
$('retry').addEventListener('click', () => { fetchModel('blaze').then(() => fetchModel('w2l')).catch(() => {}); });
$('cpu-note').textContent = navigator.gpu ? 'WebGPU / WASM' : `WASM SIMD · ${THREADS} threads`;
