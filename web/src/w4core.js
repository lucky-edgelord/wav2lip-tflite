// One engine instance with its own (non-shared) memory: used inside each Worker, or directly on the main thread as a fallback.
export function ioInfo(bytes) {
  const h = new Uint32Array(bytes.buffer, bytes.byteOffset, 16), nio = h[7], nin = h[8], o = bytes.byteOffset;
  const T = new Int32Array(bytes.buffer.slice(o + h[9], o + h[9] + h[1] * 20));
  const io = new Int32Array(bytes.buffer.slice(o + h[11], o + h[11] + nio * 4));
  return { nin, nio, sizes: Array.from(io, (t) => T[t * 5 + 1]) };
}
export class Core {
  constructor(module, memMB) {
    const pages = Math.ceil((memMB * 1048576) / 65536);
    this.memory = new WebAssembly.Memory({ initial: pages, maximum: pages });
    this.ex = new WebAssembly.Instance(module, { env: { memory: this.memory } }).exports; this.m = [];
  }
  load(slot, bytes, maxJobs) {
    const ex = this.ex, p = ex.alloc(bytes.length + 64);
    if (!p) throw new Error('out of WebAssembly memory for the model');
    new Uint8Array(this.memory.buffer, p, bytes.length).set(bytes);
    if (ex.load(slot, p, bytes.length) !== 0) throw new Error('model file rejected (bad magic)');
    const info = ioInfo(bytes);
    const ptrs = info.sizes.map((s, i) => { const q = ex.alloc(s * maxJobs * 4); if (!q) throw new Error('out of WebAssembly memory (io buffers)'); ex.set_io(slot, i, q); return q; });
    if (ex.setup_thread(slot, 0) !== 0) throw new Error('out of WebAssembly memory for the network state; raise memMB');
    this.m[slot] = { ...info, ptrs, maxJobs };
  }
  prepare(slot, inputs, n) {
    const md = this.m[slot];
    for (let i = 0; i < md.nin; i++) new Float32Array(this.memory.buffer, md.ptrs[i], n * md.sizes[i]).set(inputs[i].subarray(0, n * md.sizes[i]));
  }
  job(slot, j) { this.ex.run_job(slot, 0, j); }
  collect(slot, n) {
    const md = this.m[slot], outs = [];
    for (let i = md.nin; i < md.nio; i++) outs.push(new Float32Array(this.memory.buffer, md.ptrs[i], n * md.sizes[i]).slice());
    return outs;
  }
}
