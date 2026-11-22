// Wav2Lip in the browser: MediaPipe BlazeFace -> 96x96 crop -> Wav2Lip (LiteRT.js,
// WebGPU or CPU) -> paste back -> MP4 via WebCodecs (Mediabunny). Mirrors infer.py.
import { loadLiteRt, loadAndCompile, Tensor } from '@litertjs/core';
import { FilesetResolver, FaceDetector } from '@mediapipe/tasks-vision';
import {
  Input, ALL_FORMATS, BlobSource, CanvasSink, Output, Mp4OutputFormat, BufferTarget,
  CanvasSource, AudioBufferSource, QUALITY_HIGH, getFirstEncodableVideoCodec, getFirstEncodableAudioCodec,
} from 'mediabunny';
import { melspectrogram, melChunkStarts, writeMelChunk, SAMPLE_RATE } from './mel.js';

const IMG = 96;
const $ = (id) => document.getElementById(id);
const base = (p) => new URL(p, location.href).href;
const files = { face: null, audio: null, model: null };
let manifest = null; const modelBytes = {}; // 'gpu' | 'cpu' | 'custom' -> Uint8Array
let running = false, cancelled = false, resultBlob = null, liteRtReady = null, detector = null, detectorTs = 0;

// ---------- file pickers (click or drop) ----------
const fmtSize = (n) => (n > 1e6 ? `${(n / 1e6).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1e3))} KB`);
function setFile(kind, file, note) {
  files[kind] = file;
  const row = document.querySelector(`.pick[data-kind="${kind}"]`);
  row.classList.toggle('has', !!file);
  row.querySelector('.name').textContent = file ? `${file.name} · ${fmtSize(file.size)}${note ? ` · ${note}` : ''}` : row.dataset.empty;
  if (kind === 'face' && file) showPoster(file);
  if (kind === 'model') delete modelBytes.custom;
  $('run').disabled = !(files.face && files.audio) || running;
}
for (const row of document.querySelectorAll('.pick')) {
  const input = row.querySelector('input');
  row.dataset.empty = row.querySelector('.name').textContent;
  input.addEventListener('change', () => input.files[0] && setFile(row.dataset.kind, input.files[0]));
  row.addEventListener('dragover', (e) => { e.preventDefault(); row.classList.add('over'); });
  row.addEventListener('dragleave', () => row.classList.remove('over'));
  row.addEventListener('drop', (e) => {
    e.preventDefault(); row.classList.remove('over');
    if (e.dataTransfer.files[0]) setFile(row.dataset.kind, e.dataTransfer.files[0]);
  });
}

// ---------- bundled INT8 models, fetched from this site with progress ----------
const loading = {};
async function fetchModel(key) {
  if (modelBytes[key]) return modelBytes[key];
  return (loading[key] ??= (async () => {
    try {
      manifest ??= await (await fetch(base('./assets/models.json'))).json();
      const entry = manifest[key], url = base(`./assets/${entry.file}`);
      const label = entry.precision === 'int8' ? 'Full INT8 (CPU)' : 'INT8 weights (GPU)';
      dl({ label, url, got: 0, total: entry.bytes });
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      // The server may gzip the file, so Content-Length can be the compressed size:
      // count decompressed bytes against the size recorded in models.json.
      const reader = res.body.getReader(), chunks = [];
      let got = 0, lastPaint = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value); got += value.length;
        if (performance.now() - lastPaint > 100) { lastPaint = performance.now(); dl({ label, url, got, total: entry.bytes }); }
      }
      const buf = new Uint8Array(got);
      for (let o = 0, i = 0; i < chunks.length; o += chunks[i].length, i++) buf.set(chunks[i], o);
      if (got !== entry.bytes) throw new Error(`size mismatch: got ${fmtSize(got)}, expected ${fmtSize(entry.bytes)}`);
      modelBytes[key] = buf;
      dl({ label, url, got, total: entry.bytes, done: true });
      return buf;
    } catch (e) {
      dl({ error: `Model download failed: ${e.message}` });
      throw e;
    } finally { delete loading[key]; }
  })());
}
function dl({ label, url, got, total, done, error }) {
  const box = $('dl'), pct = total ? Math.min(100, (got / total) * 100) : 0;
  box.classList.toggle('err', !!error); box.classList.toggle('ok', !!done);
  $('retry').hidden = !error;
  if (error) { $('dl-title').textContent = error; $('model-state').textContent = 'Model: error'; return; }
  $('dl-title').textContent = done ? `${label} model ready` : `Downloading ${label} model`;
  $('dl-bar').style.width = `${pct.toFixed(1)}%`;
  $('dl-num').textContent = `${(got / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB · ${Math.floor(pct)}%`;
  $('dl-src').textContent = `from ${url.replace(/^https?:\/\//, '')}`;
  $('model-state').textContent = done ? `${label}: ready` : `${label}: ${Math.floor(pct)}%`;
  $('model-state').classList.toggle('ok', !!done);
}
const preferredKey = () => ($('accel').value === 'wasm' || !navigator.gpu ? 'cpu' : 'gpu');
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
      const wc = await new CanvasSink(track).getCanvas(await track.getFirstTimestamp());
      bmp = wc.canvas;
    }
    view.width = bmp.width; view.height = bmp.height; vctx.drawImage(bmp, 0, 0);
    stage.classList.add('ready'); $('result').hidden = true; view.hidden = false;
  } catch (e) { status(`Could not read that file: ${e.message}`); }
}
function status(msg) { $('status').textContent = msg; }
function hud(msg) { $('hud-r').textContent = msg; }
function progress(f) { $('bar').style.width = `${Math.round(f * 100)}%`; }

// ---------- runtime setup ----------
async function getDetector() {
  if (detector) return detector;
  const fileset = await FilesetResolver.forVisionTasks(base('./mediapipe'));
  for (const delegate of ['GPU', 'CPU']) {
    try {
      detector = await FaceDetector.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: base('./models/blaze_face_short_range.tflite'), delegate },
        runningMode: 'VIDEO', minDetectionConfidence: 0.5,
      });
      detector.delegateName = delegate;
      return detector;
    } catch (e) { if (delegate === 'CPU') throw e; }
  }
}

async function compile(bytes, accelerator) {
  const model = await loadAndCompile(bytes, { accelerator });
  const ins = model.getInputDetails();
  const mel = ins.find((d) => d.shape.length === 4 && d.shape[1] === 80 && d.shape[2] === 16);
  const face = ins.find((d) => d.shape.length === 4 && d.shape[3] === 6);
  if (!mel || !face) throw new Error('This does not look like a channels-last Wav2Lip model from scripts/convert.py.');
  return { model, accelerator, batch: mel.shape[0], melIndex: mel.index, faceIndex: face.index, int8: mel.dtype === 'int8' };
}

async function getModel(want) {
  liteRtReady ??= loadLiteRt(base('./litert/'));
  await liteRtReady;
  if (files.model) { // user's own model: float I/O only (no quantization params available)
    modelBytes.custom ??= new Uint8Array(await files.model.arrayBuffer());
    for (const acc of want === 'wasm' || !navigator.gpu ? ['wasm'] : ['webgpu', 'wasm']) {
      try {
        const m = await compile(modelBytes.custom, acc);
        if (m.int8) throw new Error('Custom full-INT8 models are not supported; use fp32 or int8w.');
        return m;
      } catch (e) { if (acc === 'wasm' || e.message.includes('INT8') || e.message.includes('Wav2Lip')) throw e; }
    }
  }
  const tries = want === 'wasm' || !navigator.gpu ? [['cpu', 'wasm']] : [['gpu', 'webgpu'], ['cpu', 'wasm']];
  let lastErr;
  for (const [key, acc] of tries) {
    try {
      const m = await compile(await fetchModel(key), acc);
      if (m.int8) m.q = manifest[key].io;
      m.label = key === 'gpu' ? 'INT8 weights' : 'full INT8';
      return m;
    } catch (e) { lastErr = e; status(`GPU path unavailable (${e.message}); switching to CPU…`); }
  }
  throw lastErr;
}

function quantize(src, p) {
  const out = new Int8Array(src.length), inv = 1 / p.scale;
  for (let i = 0; i < src.length; i++) out[i] = Math.max(-128, Math.min(127, Math.round(src[i] * inv) + p.zero_point));
  return out;
}

async function decodeAudio(file) {
  const buf = await file.arrayBuffer();
  const ctx = new AudioContext();
  try { var audio = await ctx.decodeAudioData(buf); } finally { ctx.close(); }
  const off = new OfflineAudioContext(1, Math.ceil(audio.duration * SAMPLE_RATE), SAMPLE_RATE);
  const src = off.createBufferSource(); src.buffer = audio; src.connect(off.destination); src.start();
  const mono16k = (await off.startRendering()).getChannelData(0);
  return { audio, mono16k };
}

// ---------- frame source: video looped to the audio length, or a still image ----------
async function openFace(file, maxW) {
  const fit = (w, h) => {
    const s = Math.min(1, maxW / w);
    return [Math.round((w * s) / 2) * 2, Math.round((h * s) / 2) * 2]; // even sizes for H.264
  };
  if (file.type.startsWith('image/')) {
    const bmp = await createImageBitmap(file);
    const [w, h] = fit(bmp.width, bmp.height);
    return { w, h, fps: null, frames: async function* (n) { for (let i = 0; i < n; i++) yield bmp; } };
  }
  const input = new Input({ formats: ALL_FORMATS, source: new BlobSource(file) });
  const track = await input.getPrimaryVideoTrack();
  if (!track) throw new Error('No video track in the face file.');
  const [w, h] = fit(track.displayWidth, track.displayHeight);
  const start = await track.getFirstTimestamp(), end = await track.computeDuration();
  const stats = await track.computePacketStats(120);
  const sink = new CanvasSink(track, { width: w, height: h, fit: 'fill' });
  return {
    w, h, fps: stats.averagePacketRate || null,
    frames: async function* (n, fps) {
      const span = Math.max(1e-3, end - start);
      const times = (function* () { for (let i = 0; i < n; i++) yield start + Math.min((i / fps) % span, span - 1e-3); })();
      let last = null;
      for await (const wc of sink.canvasesAtTimestamps(times)) { if (wc) last = wc.canvas; yield last; }
    },
  };
}

// ---------- face box: largest face, padded, EMA-smoothed (same as w2l/face.py) ----------
function boxer(padBottom, smooth) {
  let prev = null;
  return (canvas, fps, w, h) => {
    detectorTs += 1000 / fps;
    const dets = detector.detectForVideo(canvas, Math.round(detectorTs)).detections;
    if (dets.length) {
      const bb = dets.reduce((a, d) => (d.boundingBox.width * d.boundingBox.height > a.width * a.height ? d.boundingBox : a), dets[0].boundingBox);
      let box = [bb.originX, bb.originY, bb.originX + bb.width, bb.originY + bb.height * (1 + padBottom)];
      if (prev && smooth > 0) box = box.map((v, i) => smooth * prev[i] + (1 - smooth) * v);
      prev = box;
    }
    if (!prev) return null;
    const [x1, y1, x2, y2] = [Math.max(0, prev[0] | 0), Math.max(0, prev[1] | 0), Math.min(w, prev[2] | 0), Math.min(h, prev[3] | 0)];
    return x2 > x1 && y2 > y1 ? [x1, y1, x2, y2] : null;
  };
}

// ---------- main ----------
async function run() {
  running = true; cancelled = false; resultBlob = null;
  $('run').disabled = true; $('cancel').hidden = false; $('save').hidden = true; $('result').hidden = true; view.hidden = false;
  stage.classList.add('ready', 'busy'); progress(0); hud('SYNCING');
  const t = { detect: 0, model: 0, other: 0 };
  try {
    status('Loading face detector…');
    await getDetector();
    status('Loading Wav2Lip model…');
    const m = await getModel($('accel').value);
    status('Decoding audio…');
    const { audio, mono16k } = await decodeAudio(files.audio);
    const mel = await melspectrogram(mono16k, (f) => status(`Computing mel spectrogram… ${Math.round(f * 100)}%`));
    const src = await openFace(files.face, +$('maxw').value);
    const fps = +$('fps').value || (src.fps ? Math.round(src.fps * 100) / 100 : 25);
    const starts = melChunkStarts(mel.frames, fps);
    const n = starts.length, B = m.batch;
    $('acc').textContent = `Wav2Lip ${m.label} on ${m.accelerator === 'webgpu' ? 'WebGPU' : 'CPU'} · batch ${B} · face detector on ${detector.delegateName}`;

    view.width = src.w; view.height = src.h;
    hud(`${src.w}×${src.h} · ${fps} FPS`);
    const vcodec = await getFirstEncodableVideoCodec(['avc', 'vp9', 'av1'], { width: src.w, height: src.h });
    const acodec = await getFirstEncodableAudioCodec(['aac', 'opus'], { numberOfChannels: audio.numberOfChannels, sampleRate: audio.sampleRate });
    if (!vcodec) throw new Error('This browser cannot encode video (WebCodecs). Try Chrome or Edge.');
    const output = new Output({ format: new Mp4OutputFormat({ fastStart: 'in-memory' }), target: new BufferTarget() });
    const vsrc = new CanvasSource(view, { codec: vcodec, bitrate: QUALITY_HIGH });
    output.addVideoTrack(vsrc, { frameRate: fps });
    let asrc = null;
    if (acodec) { asrc = new AudioBufferSource({ codec: acodec, bitrate: QUALITY_HIGH }); output.addAudioTrack(asrc); }
    await output.start();

    const box = boxer(+$('pad').value, +$('smooth').value);
    const pool = Array.from({ length: B }, () => { const c = document.createElement('canvas'); c.width = src.w; c.height = src.h; return c; });
    const crop = new OffscreenCanvas(IMG, IMG), cctx = crop.getContext('2d', { willReadFrequently: true });
    const patch = new OffscreenCanvas(IMG, IMG), pctx = patch.getContext('2d');
    const melBuf = new Float32Array(B * 80 * 16), faceBuf = new Float32Array(B * IMG * IMG * 6);
    const frames = src.frames(n, fps);
    const t0 = performance.now();

    for (let i = 0; i < n && !cancelled; i += B) {
      const count = Math.min(B, n - i), boxes = [];
      let k = 0;
      melBuf.fill(0); faceBuf.fill(0);
      for (let j = 0; j < count; j++) {
        let s = performance.now();
        const frame = (await frames.next()).value;
        const fc = pool[j], fctx = fc.getContext('2d');
        fctx.drawImage(frame, 0, 0, src.w, src.h);
        t.other += performance.now() - s; s = performance.now();
        const b = box(fc, fps, src.w, src.h);
        t.detect += performance.now() - s; s = performance.now();
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
        t.other += performance.now() - s;
      }

      if (k) {
        const s = performance.now();
        const inputs = [];
        inputs[m.melIndex] = new Tensor(m.int8 ? quantize(melBuf, m.q.args_0) : melBuf, [B, 80, 16, 1]);
        inputs[m.faceIndex] = new Tensor(m.int8 ? quantize(faceBuf, m.q.args_1) : faceBuf, [B, IMG, IMG, 6]);
        const moved = m.accelerator === 'webgpu' ? await Promise.all(inputs.map((x) => x.moveTo('webgpu'))) : inputs;
        const outs = await m.model.run(moved);
        const host = await outs[0].moveTo('wasm');
        const raw = host.toTypedArray();
        let pred = raw.slice();
        if (m.int8) { const p = m.q.output_0; pred = Float32Array.from(raw, (v) => (v - p.zero_point) * p.scale); }
        host.delete(); moved.forEach((x) => x.delete()); outs.slice(1).forEach((x) => x.delete());
        t.model += performance.now() - s;

        const s2 = performance.now();
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
        t.other += performance.now() - s2;
      }

      const s3 = performance.now();
      for (let j = 0; j < count; j++) {
        vctx.drawImage(pool[j], 0, 0);
        await vsrc.add((i + j) / fps, 1 / fps);
      }
      t.other += performance.now() - s3;
      const done = i + count, el = (performance.now() - t0) / 1000;
      progress(done / n);
      status(`Frame ${done} of ${n}`);
      $('fps-out').textContent = (done / el).toFixed(1);
      $('ms-detect').textContent = (t.detect / done).toFixed(1);
      $('ms-model').textContent = (t.model / done).toFixed(1);
      $('ms-other').textContent = (t.other / done).toFixed(1);
    }

    if (cancelled) { await output.cancel(); status('Cancelled.'); return; }
    status('Finishing MP4…');
    if (asrc) await asrc.add(audio);
    await output.finalize();
    resultBlob = new Blob([output.target.buffer], { type: 'video/mp4' });
    const v = $('result');
    v.src = URL.createObjectURL(resultBlob); v.hidden = false; view.hidden = true;
    $('save').hidden = false;
    const secs = (performance.now() - t0) / 1000;
    status(`Done: ${n} frames in ${secs.toFixed(1)} s (${(n / secs).toFixed(1)} fps)${acodec ? '' : ' · no audio encoder, video is silent'}.`);
  } catch (e) {
    console.error(e);
    status(`Error: ${e.message || e}`);
  } finally {
    running = false; stage.classList.remove('busy');
    if (resultBlob) hud('SYNCED');
    $('cancel').hidden = true; $('run').disabled = !(files.face && files.audio);
  }
}

// ---------- save ----------
async function save() {
  if (!resultBlob) return;
  const name = `lipsync-${files.face.name.replace(/\.[^.]+$/, '')}.mp4`;
  const dl = window.claude ? await window.claude.use('downloads').catch(() => null) : null;
  if (dl) {
    try { await dl.save({ filename: name, data: resultBlob }); status('Saved.'); }
    catch (e) { status(e.code === 'declined' ? 'Save cancelled.' : `Could not save (${e.code || e.message}). Right-click the video to save it.`); }
    return;
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(resultBlob); a.download = name; a.click();
}

$('run').addEventListener('click', run);
$('cancel').addEventListener('click', () => { cancelled = true; status('Cancelling…'); });
$('save').addEventListener('click', save);
$('webgpu-note').textContent = navigator.gpu ? 'WebGPU online' : 'No WebGPU · CPU mode';
$('webgpu-note').classList.toggle('ok', !!navigator.gpu);
$('accel').addEventListener('change', () => fetchModel(preferredKey()).catch(() => {}));
$('retry').addEventListener('click', () => fetchModel(preferredKey()).catch(() => {}));
$('clear-model').addEventListener('click', (e) => { e.preventDefault(); setFile('model', null); });
