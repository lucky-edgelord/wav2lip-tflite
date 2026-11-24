// Main-thread side of the W4 engine (INT4 weights, WebAssembly SIMD). N Web Workers each own a full copy of the engine and the
// models (no SharedArrayBuffer, so no cross-origin isolation is needed: works in webxdc, GitHub Pages, file://). Frames are split
// across the workers per call. If workers cannot be created, everything runs on the main thread, yielding between frames.
import { Core, ioInfo } from './w4core.js';
const onMsg = (w, fn) => (w.on ? w.on('message', fn) : (w.onmessage = (e) => fn(e.data)));

export class W4 {
  static async create({ wasm, makeWorker, threads = 4, memMB = 96, log = () => {} }) {
    const e = new W4(); e.memMB = memMB; e.models = []; e.workers = []; e.waiting = [];
    e.module = wasm instanceof WebAssembly.Module ? wasm : await WebAssembly.compile(wasm);
    try {
      for (let t = 0; t < threads; t++) {
        const w = makeWorker(); e.workers.push(w);
        onMsg(w, (m) => e.onMessage(t, m)); w.onerror = (ev) => e.onMessage(t, { t: 'error', error: ev.message || 'worker failed to start' });
      }
      await Promise.all(e.workers.map((w, t) => e.call(t, { t: 'init', module: e.module, memMB })));
      log(`W4 engine ready: ${threads} workers, ${memMB} MB each`);
    } catch (err) {
      log(`Workers unavailable (${err.message}); running on the main thread`);
      for (const w of e.workers) try { w.terminate(); } catch {}
      e.workers = []; e.local = new Core(e.module, memMB);
    }
    e.threads = e.workers.length || 1; e.backend = 'wasm';
    return e;
  }
  onMessage(t, m) { const p = this.waiting[t]; if (!p) return; this.waiting[t] = null; m.t === 'error' ? p.rej(new Error(`worker ${t}: ${m.error}`)) : p.res(m); }
  call(t, msg, transfer) { return new Promise((res, rej) => { this.waiting[t] = { res, rej }; this.workers[t].postMessage(msg, transfer || []); }); }
  async loadModel(slot, bytes, maxJobs) {
    const info = ioInfo(bytes); this.models[slot] = { ...info, maxJobs };
    if (this.local) { this.local.load(slot, bytes, maxJobs); return info; }
    await Promise.all(this.workers.map((_, t) => this.call(t, { t: 'load', slot, bytes: bytes.slice(), maxJobs })));
    return info;
  }
  // inputs: Float32Array per model input, laid out [job][size]; returns Float32Array per output, laid out the same way
  async run(slot, inputs, n) {
    const md = this.models[slot];
    if (n > md.maxJobs) throw new Error(`too many jobs (${n} > ${md.maxJobs})`);
    if (this.local) {
      this.local.prepare(slot, inputs, n);
      for (let j = 0; j < n; j++) { this.local.job(slot, j); await new Promise((r) => setTimeout(r)); }
      return this.local.collect(slot, n);
    }
    const per = Math.ceil(n / this.workers.length), jobs = [];
    for (let t = 0, a = 0; t < this.workers.length && a < n; t++, a += per) {
      const c = Math.min(per, n - a);
      const ins = inputs.slice(0, md.nin).map((x, i) => x.slice(a * md.sizes[i], (a + c) * md.sizes[i]));
      jobs.push(this.call(t, { t: 'run', slot, inputs: ins, n: c }, ins.map((x) => x.buffer)).then((r) => ({ a, c, outputs: r.outputs })));
    }
    const parts = await Promise.all(jobs), outs = [];
    for (let k = 0; k < md.nio - md.nin; k++) {
      const o = new Float32Array(n * md.sizes[md.nin + k]);
      for (const p of parts) o.set(p.outputs[k], p.a * md.sizes[md.nin + k]);
      outs.push(o);
    }
    return outs;
  }
  terminate() { for (const w of this.workers) w.terminate(); }
}
