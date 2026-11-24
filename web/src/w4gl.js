// WebGL2 backend for the W4G1 model format: the fallback between WebGPU and the WASM CPU engine, for browsers/GPUs that have
// WebGL2 but no (working) WebGPU. Same math as the WebGPU/CPU engines: INT4 nibbles + bf16 group scales stay packed in an R32UI
// texture and are dequantized inside the fragment shader. Every tensor is a flat float array stored in an RGBA32F texture
// (4 floats per texel, 2048 texels per row), so aliased reshapes are free. Each op is one full-screen pass; every fragment
// computes 4 consecutive output floats. ConvTranspose phase convs that share an output tensor are fused into one pass.
// Same API as W4: loadModel(slot, bytes, maxJobs), run(slot, inputs, n) -> Float32Array per output.
const NF = 40, STRIDE = 52, TW = 2048, BW = 4096;

const PRE = `#version 300 es
precision highp float; precision highp int; precision highp sampler2D; precision highp usampler2D;
uniform sampler2D tA; uniform sampler2D tB; uniform usampler2D tW;
uniform ivec4 P4[52]; uniform int NG;
out vec4 o;
int P(int op, int f) { int q = op * 52 + f; return P4[q >> 2][q & 3]; }
float fa(int i) { int t = i >> 2; vec4 v = texelFetch(tA, ivec2(t & 2047, t >> 11), 0); return v[i & 3]; }
vec4 fa4(int i) { int t = i >> 2; return texelFetch(tA, ivec2(t & 2047, t >> 11), 0); }
float fb(int i) { int t = i >> 2; vec4 v = texelFetch(tB, ivec2(t & 2047, t >> 11), 0); return v[i & 3]; }
uint wd(int w) { return texelFetch(tW, ivec2(w & 4095, w >> 12), 0).r; }
float bf16(int soff, int idx) { uint w = wd((soff >> 2) + (idx >> 1)); return uintBitsToFloat(((w >> (16u * uint(idx & 1))) & 0xffffu) << 16u); }
float f32w(int off, int idx) { return uintBitsToFloat(wd((off >> 2) + idx)); }
float actf(float v, int a) { if (a == 1) return max(v, 0.0); if (a == 2) return 1.0 / (1.0 + exp(-v)); return v; }
`;
const MAIN4 = (elem) => `
float elem(int i);
void main() {
  ivec2 fc = ivec2(gl_FragCoord.xy); int i0 = (fc.y * 2048 + fc.x) * 4; int osz = P(0, 48);
  vec4 r = vec4(0.0);
  for (int j = 0; j < 4; j++) { int i = i0 + j; if (i < osz) r[j] = elem(i); }
  o = r;
}
float elem(int i) { ${elem} }
`;
const SRC = {
  conv: PRE + `
vec4 actv(int op, int oy, int ox, int k) {
  int cin = P(op, 6), kh = P(op, 8), kw = P(op, 9), sh = P(op, 10), sw = P(op, 11), ph = P(op, 12), pw = P(op, 13);
  int H = P(op, 41), W = P(op, 42), kreal = kh * kw * cin;
  if ((cin & 3) == 0) {
    if (k >= kreal) return vec4(0.0);
    int tap = k / cin; int c = k - tap * cin; int ky = tap / kw; int kx = tap - ky * kw;
    int iy = oy * sh - ph + ky, ix = ox * sw - pw + kx;
    if (iy < 0 || iy >= H || ix < 0 || ix >= W) return vec4(0.0);
    return fa4((iy * W + ix) * cin + c);
  }
  vec4 r = vec4(0.0);
  for (int l = 0; l < 4; l++) {
    int kk = k + l; if (kk >= kreal) break;
    int tap = kk / cin; int c = kk - tap * cin; int ky = tap / kw; int kx = tap - ky * kw;
    int iy = oy * sh - ph + ky, ix = ox * sw - pw + kx;
    if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
    r[l] = fa((iy * W + ix) * cin + c);
  }
  return r;
}
vec4 convN(int op, int oy, int ox, int co0, int nco) {
  int cout = P(op, 7), kpad = P(op, 20), bits = P(op, 21), woff = P(op, 22), soff = P(op, 23);
  int ng = kpad >> 5; int rowb = bits == 4 ? (kpad >> 1) : kpad;
  vec4 acc = vec4(0.0);
  for (int g = 0; g < ng; g++) {
    vec4 gs = vec4(0.0);
    for (int kk = 0; kk < 32; kk += 4) {
      vec4 a = actv(op, oy, ox, g * 32 + kk);
      for (int c = 0; c < 4; c++) {
        if (c >= nco) break;
        int co = co0 + c; vec4 wv;
        if (bits == 4) {
          uint w = wd(((woff + co * rowb + g * 16) >> 2) + ((kk & 15) >> 2)); int sh = (kk >> 4) * 4;
          wv = vec4(float(int((w >> uint(sh)) & 15u) - 8), float(int((w >> uint(sh + 8)) & 15u) - 8),
                    float(int((w >> uint(sh + 16)) & 15u) - 8), float(int((w >> uint(sh + 24)) & 15u) - 8));
        } else {
          uint w = wd((woff + co * kpad + g * 32 + kk) >> 2);
          int b0 = int(w & 255u), b1 = int((w >> 8) & 255u), b2 = int((w >> 16) & 255u), b3 = int(w >> 24);
          wv = vec4(float(b0 > 127 ? b0 - 256 : b0), float(b1 > 127 ? b1 - 256 : b1), float(b2 > 127 ? b2 - 256 : b2), float(b3 > 127 ? b3 - 256 : b3));
        }
        gs[c] += dot(a, wv);
      }
    }
    for (int c = 0; c < 4; c++) { if (c >= nco) break; acc[c] += gs[c] * bf16(soff, (co0 + c) * ng + g); }
  }
  return acc;
}
int selop(int oyF, int oxF) {
  for (int g = 0; g < NG; g++) {
    int osy = P(g, 16), osx = P(g, 17), ooy = P(g, 18), oox = P(g, 19);
    if (oyF >= ooy && oxF >= oox && (oyF - ooy) % osy == 0 && (oxF - oox) % osx == 0) return g;
  }
  return -1;
}
void main() {
  ivec2 fc = ivec2(gl_FragCoord.xy); int i0 = (fc.y * 2048 + fc.x) * 4;
  int osz = P(0, 48), cout = P(0, 7), OW = P(0, 46);
  vec4 raw = vec4(0.0); int opj[4]; opj[0] = -1; opj[1] = -1; opj[2] = -1; opj[3] = -1;
  if (i0 < osz) {
    if ((cout & 3) == 0) {
      int pix = i0 / cout; int oyF = pix / OW, oxF = pix - oyF * OW; int op = selop(oyF, oxF);
      if (op >= 0) {
        raw = convN(op, (oyF - P(op, 18)) / P(op, 16), (oxF - P(op, 19)) / P(op, 17), i0 - pix * cout, 4);
        opj[0] = op; opj[1] = op; opj[2] = op; opj[3] = op;
      }
    } else {
      for (int j = 0; j < 4; j++) {
        int i = i0 + j; if (i >= osz) break;
        int pix = i / cout; int oyF = pix / OW, oxF = pix - oyF * OW; int op = selop(oyF, oxF);
        if (op >= 0) { raw[j] = convN(op, (oyF - P(op, 18)) / P(op, 16), (oxF - P(op, 19)) / P(op, 17), i - pix * cout, 1)[0]; opj[j] = op; }
      }
    }
  }
  vec4 r = vec4(0.0);
  for (int j = 0; j < 4; j++) {
    int i = i0 + j; int op = opj[j]; if (i >= osz || op < 0) continue;
    float v = raw[j]; int co = i - (i / cout) * cout; int boff = P(op, 24);
    if (boff >= 0) v += f32w(boff, co);
    if (P(op, 5) >= 0) v += fb(i);
    r[j] = actf(v, P(op, 4));
  }
  o = r;
}`,
  dw: PRE + MAIN4(`
    int kh = P(0, 8), kw = P(0, 9), sh = P(0, 10), sw = P(0, 11), ph = P(0, 12), pw = P(0, 13);
    int woff = P(0, 22), soff = P(0, 23), boff = P(0, 24);
    int H = P(0, 41), W = P(0, 42), C = P(0, 43), OW = P(0, 46);
    int c = i % C; int pix = i / C; int ox = pix % OW, oy = pix / OW;
    float s = f32w(boff, c), sc = f32w(soff, c);
    for (int ky = 0; ky < kh; ky++) {
      int iy = oy * sh - ph + ky; if (iy < 0 || iy >= H) continue;
      for (int kx = 0; kx < kw; kx++) {
        int ix = ox * sw - pw + kx; if (ix < 0 || ix >= W) continue;
        int wi = (ky * kw + kx) * C + c; int by = woff + (wi >> 1);
        int q = int((wd(by >> 2) >> uint((by & 3) * 8 + (wi & 1) * 4)) & 15u) - 8;
        s += fa((iy * W + ix) * C + c) * float(q) * sc;
      }
    }
    return actf(s, P(0, 4));`),
  ew: PRE + MAIN4(`float v = fa(i); if (P(0, 2) >= 0) v += fb(i); return actf(v, P(0, 4));`),
  concat: PRE + MAIN4(`int c0 = P(0, 26), c1 = P(0, 27); int r = i / (c0 + c1); int k = i - r * (c0 + c1); return k < c0 ? fa(r * c0 + k) : fb(r * c1 + k - c0);`),
  pad: PRE + MAIN4(`
    int OC = P(0, 47), OW = P(0, 46); int oc = i % OC; int pix = i / OC; int ox = pix % OW, oy = pix / OW;
    int iy = oy - P(0, 25), ix = ox - P(0, 26), ic = oc - P(0, 27);
    if (iy >= 0 && iy < P(0, 41) && ix >= 0 && ix < P(0, 42) && ic >= 0 && ic < P(0, 43)) return fa((iy * P(0, 42) + ix) * P(0, 43) + ic);
    return 0.0;`),
  pool: PRE + MAIN4(`
    int C = P(0, 43), H = P(0, 41), W = P(0, 42), OW = P(0, 46);
    int c = i % C; int pix = i / C; int ox = pix % OW, oy = pix / OW; float v = -3.4e38;
    for (int ky = 0; ky < P(0, 8); ky++) { int iy = oy * P(0, 10) - P(0, 12) + ky; if (iy < 0 || iy >= H) continue;
      for (int kx = 0; kx < P(0, 9); kx++) { int ix = ox * P(0, 11) - P(0, 13) + kx; if (ix < 0 || ix >= W) continue; v = max(v, fa((iy * W + ix) * C + c)); } }
    return v;`),
};
const PROG = ['conv', 'dw', 'ew', 'concat', 'pad', 'pool', 'ew']; // OP_CONV, DW, ADD, CONCAT, PAD, POOL, RELU

export class W4GL {
  static async create({ log = () => {} } = {}) {
    const canvas = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(4, 4) : Object.assign(document.createElement('canvas'), { width: 4, height: 4 });
    const gl = canvas.getContext('webgl2', { antialias: false, depth: false, stencil: false, alpha: false, powerPreference: 'high-performance' });
    if (!gl) throw new Error('WebGL2 is not available');
    if (!gl.getExtension('EXT_color_buffer_float')) throw new Error('WebGL2 cannot render to float textures (EXT_color_buffer_float)');
    if (gl.getParameter(gl.MAX_TEXTURE_SIZE) < BW) throw new Error('WebGL2 texture size limit too small');
    const e = new W4GL(); e.gl = gl; e.models = []; e.threads = 1; e.backend = 'webgl';
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    e.name = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : 'WebGL2';
    canvas.addEventListener?.('webglcontextlost', (ev) => { ev.preventDefault(); e.lost = true; log('WebGL context lost'); });
    const vs = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vs, `#version 300 es\nvoid main() { vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }`);
    gl.compileShader(vs);
    e.prog = {};
    for (const n of new Set(PROG)) {
      const fs = gl.createShader(gl.FRAGMENT_SHADER); gl.shaderSource(fs, SRC[n]); gl.compileShader(fs);
      const pr = gl.createProgram(); gl.attachShader(pr, vs); gl.attachShader(pr, fs); gl.linkProgram(pr);
      if (!gl.getProgramParameter(pr, gl.LINK_STATUS)) throw new Error(`GLSL ${n}: ${gl.getShaderInfoLog(fs) || gl.getProgramInfoLog(pr)}`);
      e.prog[n] = { pr, u: { tA: gl.getUniformLocation(pr, 'tA'), tB: gl.getUniformLocation(pr, 'tB'), tW: gl.getUniformLocation(pr, 'tW'), P4: gl.getUniformLocation(pr, 'P4'), NG: gl.getUniformLocation(pr, 'NG') } };
    }
    e.fbo = gl.createFramebuffer();
    e.dummy = e.newTex(4);
    log(`WebGL2 ready: ${e.name}`);
    return e;
  }

  newTex(floats) {
    const gl = this.gl, texels = Math.max(1, Math.ceil(floats / 4)), h = Math.ceil(texels / TW), tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex); gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32F, TW, h);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    return { tex, h, size: floats };
  }

  async loadModel(slot, bytes, maxJobs) {
    const gl = this.gl, h = new Uint32Array(bytes.buffer, bytes.byteOffset, 16), o0 = bytes.byteOffset;
    if (h[0] !== 0x31473457) throw new Error('model file rejected (bad magic)');
    const nt = h[1], no = h[2], nio = h[7], nin = h[8];
    const T = new Int32Array(bytes.buffer.slice(o0 + h[9], o0 + h[9] + nt * 20));
    const O = new Int32Array(bytes.buffer.slice(o0 + h[10], o0 + h[10] + no * NF * 4));
    const IO = new Int32Array(bytes.buffer.slice(o0 + h[11], o0 + h[11] + nio * 4));
    // blob -> R32UI texture, 4096 words per row
    const blobLen = bytes.length - h[12], words = (blobLen + 3) >> 2, rows = Math.ceil(words / BW);
    const wtex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, wtex); gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R32UI, BW, rows);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
    const bb = (o0 + h[12]) % 4 === 0 ? new Uint32Array(bytes.buffer, o0 + h[12], blobLen >> 2) : new Uint32Array(bytes.slice(h[12]).buffer, 0, blobLen >> 2);
    const full = Math.floor(bb.length / BW);
    if (full) gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, BW, full, gl.RED_INTEGER, gl.UNSIGNED_INT, bb.subarray(0, full * BW));
    if (bb.length > full * BW) { const tail = new Uint32Array(BW); tail.set(bb.subarray(full * BW)); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, full, BW, 1, gl.RED_INTEGER, gl.UNSIGNED_INT, tail); }
    // resolve tensors -> textures statically: a read gets the latest texture written at that arena offset (reshapes alias)
    const cur = new Map(), tOff = (t) => T[t * 5], tSz = (t) => T[t * 5 + 1];
    const get = (t) => { const s = cur.get(tOff(t)); if (!s || s.size !== tSz(t)) throw new Error(`tensor ${t} has no matching producer`); return s; };
    const inputs = []; for (let i = 0; i < nin; i++) { const t = this.newTex(tSz(IO[i])); cur.set(tOff(IO[i]), t); inputs.push(t); }
    const groups = [];
    for (let i = 0; i < no;) {
      const row = (k) => O.subarray(k * NF, (k + 1) * NF);
      let n = 1; if (row(i)[0] === 0) while (n < 4 && i + n < no && row(i + n)[0] === 0 && row(i + n)[3] === row(i)[3]) n++;
      const P4 = new Int32Array(4 * STRIDE);
      for (let g = 0; g < n; g++) {
        const r = row(i + g), u = P4.subarray(g * STRIDE, (g + 1) * STRIDE); u.set(r);
        const ti = r[1] >= 0 ? T.subarray(r[1] * 5, r[1] * 5 + 5) : [0, 0, 0, 0, 0], to = T.subarray(r[3] * 5, r[3] * 5 + 5);
        u[41] = ti[2]; u[42] = ti[3]; u[43] = ti[4]; u[45] = to[2]; u[46] = to[3]; u[47] = to[4]; u[48] = to[1];
        if (r[0] === 0 && (r[22] % 4 || r[23] % 4)) throw new Error('unaligned weights');
      }
      const r0 = row(i), a = r0[1] >= 0 ? get(r0[1]) : this.dummy;
      let b = this.dummy; if (r0[0] === 0) { if (r0[5] >= 0) b = get(r0[5]); } else if (r0[2] >= 0) b = get(r0[2]);
      const out = this.newTex(tSz(r0[3])); cur.set(tOff(r0[3]), out);
      groups.push({ prog: PROG[r0[0]], P4, ng: n, a, b, out });
      i += n;
    }
    const outs = []; for (let i = nin; i < nio; i++) outs.push(get(IO[i]));
    this.models[slot] = { nin, nio, sizes: Array.from(IO, (t) => tSz(t)), inputs, outs, groups, wtex };
    return { nin, nio, sizes: this.models[slot].sizes };
  }

  frame(m, inputs, j) {
    const gl = this.gl;
    for (let i = 0; i < m.nin; i++) {
      const t = m.inputs[i], sz = m.sizes[i], buf = new Float32Array(TW * t.h * 4);
      buf.set(inputs[i].subarray(j * sz, (j + 1) * sz));
      gl.bindTexture(gl.TEXTURE_2D, t.tex); gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TW, t.h, gl.RGBA, gl.FLOAT, buf);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo); gl.disable(gl.BLEND); gl.disable(gl.DEPTH_TEST);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, m.wtex);
    let last = null;
    for (const g of m.groups) {
      const pr = this.prog[g.prog]; if (last !== pr) { gl.useProgram(pr.pr); gl.uniform1i(pr.u.tA, 0); gl.uniform1i(pr.u.tB, 1); gl.uniform1i(pr.u.tW, 2); last = pr; }
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, g.a.tex);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, g.b.tex);
      gl.uniform4iv(pr.u.P4, g.P4); gl.uniform1i(pr.u.NG, g.ng);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, g.out.tex, 0);
      gl.viewport(0, 0, TW, g.out.h); gl.drawArrays(gl.TRIANGLES, 0, 3);
    }
  }

  async run(slot, inputs, n) {
    const gl = this.gl, m = this.models[slot], outs = [];
    for (let k = m.nin; k < m.nio; k++) outs.push(new Float32Array(n * m.sizes[k]));
    for (let j = 0; j < n; j++) {
      if (this.lost) throw new Error('WebGL context lost');
      this.frame(m, inputs, j);
      for (let k = 0; k < m.outs.length; k++) {
        const t = m.outs[k], buf = new Float32Array(TW * t.h * 4);
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t.tex, 0);
        gl.readPixels(0, 0, TW, t.h, gl.RGBA, gl.FLOAT, buf);
        outs[k].set(buf.subarray(0, t.size), j * t.size);
      }
      if (gl.getError() !== gl.NO_ERROR && this.lost) throw new Error('WebGL error');
      await new Promise((r) => setTimeout(r));
    }
    return outs;
  }
}
