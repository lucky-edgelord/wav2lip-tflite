// WebGPU backend for the W4G1 model format (same files as the WASM engine). Approach follows llama.cpp's WebGPU backend:
// weights stay packed in a storage buffer (INT4 nibbles + bf16 group scales, 8 nibbles per u32) and are dequantized inside the
// matmul kernel while it fills its shared-memory tile; activations stay f32. Convolutions are implicit-GEMM (no im2col buffer),
// 64x64 output tile per workgroup, 16 k-values per step. Whole frames are batched along dispatch z so the GPU stays busy.
// Same API as W4 (w4.js): loadModel(slot, bytes, maxJobs), run(slot, inputs, n) -> Float32Array per output.
const NF = 40, SLOT = 256; // op row ints, bytes per op in the uniform buffer
const X = { IN_OFF: 40, IN_H: 41, IN_W: 42, IN_C: 43, OUT_OFF: 44, OUT_H: 45, OUT_W: 46, OUT_C: 47, OUT_N: 48, RES_OFF: 49, IN1_OFF: 50, STRIDE: 51 };

const WGSL = /* wgsl */ `
struct P { v: array<vec4<i32>, 16> };
@group(0) @binding(0) var<storage, read> blob: array<u32>;
@group(0) @binding(1) var<storage, read_write> arena: array<f32>;
@group(0) @binding(2) var<uniform> prm: P;
fn p(i: u32) -> i32 { return prm.v[i >> 2u][i & 3u]; }
fn act_f(v: f32, a: i32) -> f32 {
  if (a == 1) { return max(v, 0.0); }
  if (a == 2) { return 1.0 / (1.0 + exp(-v)); }
  return v;
}
fn bf16(idx: u32, soff: i32) -> f32 {
  let w = blob[(u32(soff) >> 2u) + (idx >> 1u)];
  return bitcast<f32>(((w >> (16u * (idx & 1u))) & 0xffffu) << 16u);
}
fn gid1(wg: vec3<u32>, lid: u32) -> u32 { return (wg.y * 65535u + wg.x) * 64u + lid; }

var<workgroup> As: array<f32, 1024>;
var<workgroup> Ws: array<f32, 1024>;

@compute @workgroup_size(256)
fn conv(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let cin = p(6u); let cout = p(7u); let kh = p(8u); let kw = p(9u); let sh = p(10u); let sw = p(11u);
  let ph = p(12u); let pw = p(13u); let hl = p(14u); let wl = p(15u);
  let osy = p(16u); let osx = p(17u); let ooy = p(18u); let oox = p(19u);
  let kpad = p(20u); let bits = p(21u); let woff = p(22u); let soff = p(23u); let boff = p(24u); let act = p(4u);
  let H = p(41u); let W = p(42u); let OW = p(46u);
  let base = i32(wg.z) * p(51u);
  let inb = base + p(40u); let outb = base + p(44u); let resb = p(49u);
  let P = hl * wl; let ng = kpad >> 5u;
  let p0 = i32(wg.x) * 64; let co0 = i32(wg.y) * 64;
  let kreal = kh * kw * cin;
  let rowb = select(kpad, kpad >> 1u, bits == 4);
  let tx = i32(tid & 15u); let ty = i32(tid >> 4u);
  var acc: array<f32, 16>;
  for (var i = 0; i < 16; i++) { acc[i] = 0.0; }
  for (var k0 = 0; k0 < kpad; k0 += 16) {
    // A tile: 64 pixels x 16 k (gathered straight from the NHWC input)
    let kk = i32(tid & 15u); let pr = i32(tid >> 4u);
    let k = k0 + kk;
    let tap = k / cin; let c = k - tap * cin; let ky = tap / kw; let kx = tap - ky * kw;
    for (var i = 0; i < 4; i++) {
      let m = pr + 16 * i; let pp = p0 + m;
      var v = 0.0;
      if (k < kreal && pp < P) {
        let oy = pp / wl; let ox = pp - oy * wl;
        let iy = oy * sh - ph + ky; let ix = ox * sw - pw + kx;
        if (iy >= 0 && iy < H && ix >= 0 && ix < W) { v = arena[inb + (iy * W + ix) * cin + c]; }
      }
      As[kk * 64 + m] = v;
    }
    // W tile: 64 output channels x 16 k, dequantized from the packed blob (one u32 = 4 k-values per thread)
    let cl = i32(tid >> 2u); let j = i32(tid & 3u); let co = co0 + cl;
    if (co < cout) {
      if (bits == 4) {
        let bo = woff + co * rowb + (k0 >> 5u) * 16;
        let w = blob[u32((bo >> 2u) + j)];
        let sc = bf16(u32(co * ng + (k0 >> 5u)), soff);
        let sh4 = u32(((k0 >> 4u) & 1) * 4);
        for (var t = 0; t < 4; t++) {
          let q = i32((w >> (u32(t * 8) + sh4)) & 15u) - 8;
          Ws[(j * 4 + t) * 64 + cl] = f32(q) * sc;
        }
      } else {
        let bo = woff + co * rowb + k0;
        let w = blob[u32((bo >> 2u) + j)];
        let sc = bf16(u32(co * ng + (k0 >> 5u)), soff);
        for (var t = 0; t < 4; t++) {
          let q = extractBits(bitcast<i32>(w), u32(t * 8), 8u);
          Ws[(j * 4 + t) * 64 + cl] = f32(q) * sc;
        }
      }
    } else {
      for (var t = 0; t < 4; t++) { Ws[(j * 4 + t) * 64 + cl] = 0.0; }
    }
    workgroupBarrier();
    for (var q = 0; q < 16; q++) {
      var a: array<f32, 4>; var b: array<f32, 4>;
      for (var i = 0; i < 4; i++) { a[i] = As[q * 64 + ty + 16 * i]; b[i] = Ws[q * 64 + tx + 16 * i]; }
      for (var i = 0; i < 4; i++) { for (var jj = 0; jj < 4; jj++) { acc[i * 4 + jj] += a[i] * b[jj]; } }
    }
    workgroupBarrier();
  }
  for (var i = 0; i < 4; i++) {
    let pp = p0 + ty + 16 * i;
    if (pp >= P) { continue; }
    let oy = pp / wl; let ox = pp - oy * wl;
    let pix = ((oy * osy + ooy) * OW + (ox * osx + oox)) * cout;
    for (var jj = 0; jj < 4; jj++) {
      let co = co0 + tx + 16 * jj;
      if (co >= cout) { continue; }
      var v = acc[i * 4 + jj];
      if (boff >= 0) { v += bitcast<f32>(blob[u32((boff >> 2u) + co)]); }
      if (resb >= 0) { v += arena[base + resb + pix + co]; }
      arena[outb + pix + co] = act_f(v, act);
    }
  }
}

@compute @workgroup_size(64)
fn dw(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let idx = i32(gid1(wg, lid)); let n = p(48u);
  if (idx >= n) { return; }
  let kh = p(8u); let kw = p(9u); let sh = p(10u); let sw = p(11u); let ph = p(12u); let pw = p(13u);
  let woff = p(22u); let soff = p(23u); let boff = p(24u); let act = p(4u);
  let H = p(41u); let W = p(42u); let C = p(43u); let OW = p(46u);
  let base = i32(wg.z) * p(51u); let inb = base + p(40u);
  let c = idx % C; let pix = idx / C; let ox = pix % OW; let oy = pix / OW;
  var s = bitcast<f32>(blob[u32((boff >> 2u) + c)]);
  let sc = bitcast<f32>(blob[u32((soff >> 2u) + c)]);
  for (var ky = 0; ky < kh; ky++) {
    let iy = oy * sh - ph + ky; if (iy < 0 || iy >= H) { continue; }
    for (var kx = 0; kx < kw; kx++) {
      let ix = ox * sw - pw + kx; if (ix < 0 || ix >= W) { continue; }
      let wi = (ky * kw + kx) * C + c;
      let by = woff + (wi >> 1u);
      let q = i32((blob[u32(by >> 2u)] >> (u32((by & 3) * 8) + u32((wi & 1) * 4))) & 15u) - 8;
      s += arena[inb + (iy * W + ix) * C + c] * f32(q) * sc;
    }
  }
  arena[base + p(44u) + idx] = act_f(s, act);
}

@compute @workgroup_size(64)
fn ew(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let idx = i32(gid1(wg, lid)); if (idx >= p(48u)) { return; }
  let base = i32(wg.z) * p(51u);
  var v = arena[base + p(40u) + idx];
  if (p(2u) >= 0) { v += arena[base + p(50u) + idx]; }
  arena[base + p(44u) + idx] = act_f(v, p(4u));
}

@compute @workgroup_size(64)
fn concat(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let idx = i32(gid1(wg, lid)); if (idx >= p(48u)) { return; }
  let base = i32(wg.z) * p(51u);
  let c0 = p(26u); let c1 = p(27u);
  let r = idx / (c0 + c1); let k = idx - r * (c0 + c1);
  var v: f32;
  if (k < c0) { v = arena[base + p(40u) + r * c0 + k]; } else { v = arena[base + p(50u) + r * c1 + k - c0]; }
  arena[base + p(44u) + idx] = v;
}

@compute @workgroup_size(64)
fn pad(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let idx = i32(gid1(wg, lid)); if (idx >= p(48u)) { return; }
  let base = i32(wg.z) * p(51u);
  let OC = p(47u); let OW = p(46u);
  let oc = idx % OC; let pix = idx / OC; let ox = pix % OW; let oy = pix / OW;
  let iy = oy - p(25u); let ix = ox - p(26u); let ic = oc - p(27u);
  var v = 0.0;
  if (iy >= 0 && iy < p(41u) && ix >= 0 && ix < p(42u) && ic >= 0 && ic < p(43u)) {
    v = arena[base + p(40u) + (iy * p(42u) + ix) * p(43u) + ic];
  }
  arena[base + p(44u) + idx] = v;
}

@compute @workgroup_size(64)
fn pool(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wg: vec3<u32>) {
  let idx = i32(gid1(wg, lid)); if (idx >= p(48u)) { return; }
  let base = i32(wg.z) * p(51u);
  let C = p(43u); let H = p(41u); let W = p(42u); let OW = p(46u);
  let c = idx % C; let pix = idx / C; let ox = pix % OW; let oy = pix / OW;
  var v = -3.4e38;
  for (var ky = 0; ky < p(8u); ky++) {
    let iy = oy * p(10u) - p(12u) + ky; if (iy < 0 || iy >= H) { continue; }
    for (var kx = 0; kx < p(9u); kx++) {
      let ix = ox * p(11u) - p(13u) + kx; if (ix < 0 || ix >= W) { continue; }
      v = max(v, arena[base + p(40u) + (iy * W + ix) * C + c]);
    }
  }
  arena[base + p(44u) + idx] = v;
}
`;

const OPS = ['conv', 'dw', 'ew', 'concat', 'pad', 'pool', 'ew']; // OP_CONV, DW, ADD, CONCAT, PAD, POOL, RELU

export class W4GPU {
  static async create({ log = () => {}, powerPreference = 'high-performance' } = {}) {
    if (typeof navigator === 'undefined' || !navigator.gpu) throw new Error('WebGPU is not available in this browser');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference });
    if (!adapter) throw new Error('no WebGPU adapter');
    const want = {}; for (const k of ['maxStorageBufferBindingSize', 'maxBufferSize']) want[k] = Math.min(adapter.limits[k], 1 << 30);
    const device = await adapter.requestDevice({ requiredLimits: want });
    const e = new W4GPU(); e.device = device; e.limits = want; e.models = [];
    const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
    e.name = [info.vendor, info.architecture, info.description].filter(Boolean).join(' ') || 'WebGPU adapter';
    device.lost.then((l) => log(`WebGPU device lost: ${l.message}`));
    device.addEventListener?.('uncapturederror', (ev) => log(`WebGPU error: ${ev.error.message}`));
    const module = device.createShaderModule({ code: WGSL });
    const info2 = await module.getCompilationInfo();
    for (const m of info2.messages) if (m.type === 'error') throw new Error(`WGSL: ${m.message} (line ${m.lineNum})`);
    e.bgl = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true } },
    ] });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [e.bgl] });
    e.pipe = {};
    for (const n of new Set(OPS)) e.pipe[n] = await device.createComputePipelineAsync({ layout, compute: { module, entryPoint: n } });
    e.threads = 1; e.backend = 'webgpu';
    log(`WebGPU ready: ${e.name}`);
    return e;
  }

  async loadModel(slot, bytes, maxJobs) {
    const d = this.device, h = new Uint32Array(bytes.buffer, bytes.byteOffset, 16), o0 = bytes.byteOffset;
    if (h[0] !== 0x31473457) throw new Error('model file rejected (bad magic)');
    const nt = h[1], no = h[2], arenaF = h[3], nio = h[7], nin = h[8];
    const T = new Int32Array(bytes.buffer.slice(o0 + h[9], o0 + h[9] + nt * 20));
    const O = new Int32Array(bytes.buffer.slice(o0 + h[10], o0 + h[10] + no * NF * 4));
    const IO = new Int32Array(bytes.buffer.slice(o0 + h[11], o0 + h[11] + nio * 4));
    const blobLen = bytes.length - h[12], blobBytes = new Uint8Array(bytes.buffer, o0 + h[12], blobLen);
    const blob = d.createBuffer({ size: (blobLen + 3) & ~3, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Uint8Array(blob.getMappedRange()).set(blobBytes); blob.unmap();
    const stride = arenaF, perBatch = stride * 4;
    const B = Math.max(1, Math.min(maxJobs, Math.floor(this.limits.maxStorageBufferBindingSize / perBatch)));
    const arena = d.createBuffer({ size: perBatch * B, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const ub = new Int32Array(no * SLOT / 4);
    const ops = [];
    for (let i = 0; i < no; i++) {
      const row = O.subarray(i * NF, (i + 1) * NF), u = ub.subarray(i * SLOT / 4, (i + 1) * SLOT / 4);
      u.set(row);
      const ti = row[1] >= 0 ? T.subarray(row[1] * 5, row[1] * 5 + 5) : [0, 0, 0, 0, 0], to = T.subarray(row[3] * 5, row[3] * 5 + 5);
      u[X.IN_OFF] = ti[0]; u[X.IN_H] = ti[2]; u[X.IN_W] = ti[3]; u[X.IN_C] = ti[4];
      u[X.OUT_OFF] = to[0]; u[X.OUT_H] = to[2]; u[X.OUT_W] = to[3]; u[X.OUT_C] = to[4]; u[X.OUT_N] = to[1];
      u[X.RES_OFF] = row[5] >= 0 ? T[row[5] * 5] : -1;
      u[X.IN1_OFF] = row[2] >= 0 ? T[row[2] * 5] : 0; u[X.STRIDE] = stride;
      if (row[0] === 0 && (row[22] % 4 || row[23] % 4)) throw new Error('unaligned weights');
      const t = OPS[row[0]], P = row[0] === 0 ? Math.ceil(row[14] * row[15] / 64) : Math.ceil(to[1] / 64);
      ops.push({ t, x: row[0] === 0 ? P : Math.min(P, 65535), y: row[0] === 0 ? Math.ceil(row[7] / 64) : Math.ceil(P / 65535), off: i * SLOT });
    }
    const uni = d.createBuffer({ size: ub.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(uni, 0, ub);
    const bg = d.createBindGroup({ layout: this.bgl, entries: [
      { binding: 0, resource: { buffer: blob } }, { binding: 1, resource: { buffer: arena } }, { binding: 2, resource: { buffer: uni, size: SLOT } } ] });
    const sizes = Array.from(IO, (t) => T[t * 5 + 1]), offs = Array.from(IO, (t) => T[t * 5]);
    this.models[slot] = { nin, nio, sizes, offs, B, stride, arena, bg, ops, maxJobs };
    return { nin, nio, sizes };
  }

  async run(slot, inputs, n) {
    const d = this.device, m = this.models[slot], outs = [];
    for (let k = m.nin; k < m.nio; k++) outs.push(new Float32Array(n * m.sizes[k]));
    for (let a = 0; a < n; a += m.B) {
      const c = Math.min(m.B, n - a);
      for (let b = 0; b < c; b++) for (let i = 0; i < m.nin; i++) {
        const sz = m.sizes[i], src = inputs[i].subarray((a + b) * sz, (a + b + 1) * sz);
        d.queue.writeBuffer(m.arena, (b * m.stride + m.offs[i]) * 4, src.buffer, src.byteOffset, sz * 4);
      }
      const enc = d.createCommandEncoder(), pass = enc.beginComputePass();
      for (const op of m.ops) { pass.setPipeline(this.pipe[op.t]); pass.setBindGroup(0, m.bg, [op.off]); pass.dispatchWorkgroups(op.x, op.y, c); }
      pass.end();
      const stg = [];
      for (let k = m.nin; k < m.nio; k++) {
        const sz = m.sizes[k], s = d.createBuffer({ size: sz * 4 * c, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        for (let b = 0; b < c; b++) enc.copyBufferToBuffer(m.arena, (b * m.stride + m.offs[k]) * 4, s, b * sz * 4, sz * 4);
        stg.push(s);
      }
      d.queue.submit([enc.finish()]);
      for (let k = 0; k < stg.length; k++) {
        await stg[k].mapAsync(GPUMapMode.READ);
        outs[k].set(new Float32Array(stg[k].getMappedRange()), a * m.sizes[m.nin + k]); stg[k].destroy();
      }
    }
    return outs;
  }
}
