// W4 engine: INT4-weight / INT8-activation CNN runtime for WebAssembly (SIMD128 + shared memory).
// Freestanding C (no libc). Weights are stored as packed nibbles (group of 32 along K, bf16 scale per
// group); activations are float between layers and quantized per tensor to 8 bits right before each
// conv; the dot products run on i16x8 lanes with i32 accumulation. One frame = one job, so N workers
// sharing this memory run N frames in parallel without any locking inside the network.
#include <stdint.h>
#include <stddef.h>
#ifdef __wasm__
#include <wasm_simd128.h>
#define EXPORT(n) __attribute__((export_name(n)))
#else  // native x86-64 build (Colab / desktop): same kernel, SSE4.1 instead of WebAssembly SIMD
#include <immintrin.h>
#include <stdlib.h>
#define EXPORT(n)
typedef __m128i v128_t;
#define wasm_i32x4_splat(x) _mm_set1_epi32(x)
#define wasm_f32x4_splat(x) _mm_castps_si128(_mm_set1_ps(x))
#define wasm_v128_load(p) _mm_loadu_si128((const __m128i *)(p))
#define wasm_v128_store(p, v) _mm_storeu_si128((__m128i *)(p), v)
#define wasm_v128_and _mm_and_si128
#define wasm_u8x16_splat(x) _mm_set1_epi8((char)(x))
#define wasm_i8x16_splat(x) _mm_set1_epi8((char)(x))
#define wasm_i8x16_sub _mm_sub_epi8
#define wasm_u8x16_shr(v, n) _mm_and_si128(_mm_srli_epi16(v, n), _mm_set1_epi8(0x0F))
#define wasm_i16x8_extend_low_i8x16(v) _mm_cvtepi8_epi16(v)
#define wasm_i16x8_extend_high_i8x16(v) _mm_cvtepi8_epi16(_mm_srli_si128(v, 8))
#define wasm_i32x4_dot_i16x8 _mm_madd_epi16
#define wasm_i32x4_add _mm_add_epi32
#define wasm_f32x4_add(a, b) _mm_castps_si128(_mm_add_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b)))
#define wasm_f32x4_mul(a, b) _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b)))
#define wasm_f32x4_convert_i32x4(a) _mm_castps_si128(_mm_cvtepi32_ps(a))
static inline float wasm_f32x4_extract_lane(v128_t v, int i) { float f[4]; _mm_storeu_ps(f, _mm_castsi128_ps(v)); return f[i]; }
#endif
#define MAXT 8
#define NF 40
typedef int32_t i32; typedef uint32_t u32; typedef int16_t i16;

enum { OP_CONV, OP_DW, OP_ADD, OP_CONCAT, OP_PAD, OP_POOL, OP_RELU };
enum { F_TYPE, F_IN0, F_IN1, F_OUT, F_ACT, F_RES, F_CIN, F_COUT, F_KH, F_KW, F_SH, F_SW, F_PH, F_PW, F_HL, F_WL,
       F_OSY, F_OSX, F_OOY, F_OOX, F_KPAD, F_WBITS, F_WOFF, F_SOFF, F_BOFF, F_X0, F_X1, F_X2, F_X3 };

typedef struct { const i32 *T, *O, *IO; const uint8_t *blob; i32 nt, no, nio, nin, arena, max_qin, max_col, max_wrow, abits; } Model;
typedef struct { float *arena; i16 *qin, *col, *wrow; float *wf; } Ctx;
static Model M[2];
static Ctx CX[2][MAXT];
static float *g_io[2][8];
static volatile i32 g_next;

#ifdef __wasm__
extern unsigned char __heap_base;
static size_t g_brk;
EXPORT("alloc") size_t w_alloc(size_t n) {
  if (!g_brk) g_brk = (size_t)&__heap_base;
  g_brk = (g_brk + 63) & ~(size_t)63;
  size_t p = g_brk; g_brk += n;
  if (g_brk > (size_t)__builtin_wasm_memory_size(0) * 65536u) { g_brk = p; return 0; }
  return p;
}
#else
size_t w_alloc(size_t n) { void *p = aligned_alloc(64, (n + 63) & ~(size_t)63); return (size_t)p; }
#endif
EXPORT("reset_jobs") void reset_jobs(void) { g_next = 0; }
EXPORT("take_job") i32 take_job(void) { return __atomic_fetch_add(&g_next, 1, __ATOMIC_SEQ_CST); }

EXPORT("load") i32 load(i32 m, const uint8_t *p, i32 len) {
  const u32 *h = (const u32 *)p;
  if (len < 64 || h[0] != 0x31473457u) return -1;       // "W4G1"
  Model *d = &M[m];
  d->nt = h[1]; d->no = h[2]; d->arena = h[3]; d->max_qin = h[4]; d->max_col = h[5]; d->max_wrow = h[6];
  d->nio = h[7]; d->nin = h[8]; d->abits = h[13] ? (i32)h[13] : 8;
  d->T = (const i32 *)(p + h[9]); d->O = (const i32 *)(p + h[10]); d->IO = (const i32 *)(p + h[11]); d->blob = p + h[12];
  return 0;
}
EXPORT("set_io") void set_io(i32 m, i32 i, float *p) { g_io[m][i] = p; }
EXPORT("setup_thread") i32 setup_thread(i32 m, i32 t) {
  Model *d = &M[m]; Ctx *c = &CX[m][t];
  c->arena = (float *)w_alloc((size_t)d->arena * 4);
  c->qin = (i16 *)w_alloc(((size_t)d->max_qin + 8192) * 2);
  c->col = (i16 *)w_alloc(((size_t)d->max_col + 64) * 2);
  c->wrow = (i16 *)w_alloc(((size_t)d->max_wrow * 2 + 64) * 2);
  c->wf = (float *)w_alloc(((size_t)d->max_wrow + 64) * 4);
  return (c->arena && c->qin && c->col && c->wrow && c->wf) ? 0 : -1;
}

static inline float fexp(float x) {
  if (x < -87.f) x = -87.f; if (x > 88.f) x = 88.f;
  float t = x * 1.442695041f, fl = __builtin_floorf(t), f = t - fl;
  float p = 1.f + f * (0.6931472f + f * (0.2402265f + f * (0.0555041f + f * (0.0096181f + f * 0.0013333f))));
  union { u32 u; float f; } b; b.u = (u32)((i32)fl + 127) << 23;
  return p * b.f;
}
static inline float act_f(float v, i32 a) {
  if (a == 1) return v < 0.f ? 0.f : v;
  if (a == 2) return 1.f / (1.f + fexp(-v));
  return v;
}
static inline float bf16(uint16_t b) { union { u32 u; float f; } x; x.u = (u32)b << 16; return x.f; }
static inline float hsum(v128_t v) {
  return wasm_f32x4_extract_lane(v, 0) + wasm_f32x4_extract_lane(v, 1) + wasm_f32x4_extract_lane(v, 2) + wasm_f32x4_extract_lane(v, 3);
}

// Unpack one weight row (kpad values) to i16. 4-bit: byte j of a 16-byte group = elem j (lo) | elem j+16 (hi), offset 8.
static void unpack_row(const uint8_t *q, i32 kpad, i32 bits, i16 *dst) {
  if (bits == 4) {
    const v128_t m = wasm_u8x16_splat(15), e = wasm_i8x16_splat(8);
    for (i32 g = 0; g < kpad; g += 32) {
      v128_t v = wasm_v128_load(q + (g >> 1));
      v128_t lo = wasm_i8x16_sub(wasm_v128_and(v, m), e), hi = wasm_i8x16_sub(wasm_u8x16_shr(v, 4), e);
      wasm_v128_store(dst + g, wasm_i16x8_extend_low_i8x16(lo));
      wasm_v128_store(dst + g + 8, wasm_i16x8_extend_high_i8x16(lo));
      wasm_v128_store(dst + g + 16, wasm_i16x8_extend_low_i8x16(hi));
      wasm_v128_store(dst + g + 24, wasm_i16x8_extend_high_i8x16(hi));
    }
  } else {
    for (i32 k = 0; k < kpad; k += 16) {
      v128_t v = wasm_v128_load(q + k);
      wasm_v128_store(dst + k, wasm_i16x8_extend_low_i8x16(v));
      wasm_v128_store(dst + k + 8, wasm_i16x8_extend_high_i8x16(v));
    }
  }
}

#define DOT(a, w) wasm_i32x4_dot_i16x8(a, w)
static void op_conv(const Model *md, Ctx *c, const i32 *o) {
  float *ar = c->arena;
  const i32 *ti = md->T + 5 * o[F_IN0], *to = md->T + 5 * o[F_OUT];
  const float *x = ar + ti[0];
  const i32 H = ti[2], W = ti[3], cin = o[F_CIN], cout = o[F_COUT], kh = o[F_KH], kw = o[F_KW], sh = o[F_SH], sw = o[F_SW];
  const i32 ph = o[F_PH], pw = o[F_PW], hl = o[F_HL], wl = o[F_WL], osy = o[F_OSY], osx = o[F_OSX], ooy = o[F_OOY], oox = o[F_OOX];
  const i32 kpad = o[F_KPAD], ng = kpad >> 5, bits = o[F_WBITS], act = o[F_ACT], OW = to[3];
  float *out = ar + to[0];
  const float *res = o[F_RES] >= 0 ? ar + md->T[5 * o[F_RES]] : 0;
  const uint8_t *wq = md->blob + o[F_WOFF];
  const uint16_t *ws = (const uint16_t *)(md->blob + o[F_SOFF]);
  const float *bias = o[F_BOFF] >= 0 ? (const float *)(md->blob + o[F_BOFF]) : 0;
  const size_t rowb = bits == 4 ? (size_t)kpad >> 1 : (size_t)kpad;

  // dynamic per-tensor activation quantization to 8 bits (unsigned range if the tensor is non-negative)
  const i32 n = H * W * cin;
  float mx = 0.f, mn = 0.f;
  for (i32 i = 0; i < n; i++) { float v = x[i]; mx = v > mx ? v : mx; mn = v < mn ? v : mn; }
  float amax = mx > -mn ? mx : -mn;
  const float qmax = md->abits == 4 ? (mn >= 0.f ? 15.f : 7.f) : (mn >= 0.f ? 255.f : 127.f);
  const float inv = amax > 0.f ? qmax / amax : 0.f, sa = amax > 0.f ? amax / qmax : 0.f;
  i16 *qin = c->qin;
  for (i32 i = 0; i < n; i++) { float v = x[i] * inv; qin[i] = (i16)(v + (v < 0.f ? -0.5f : 0.5f)); }
  for (i32 i = n; i < n + 8192 && i < n + 4 * cin; i++) qin[i] = 0;

  const i32 direct = (kh == 1 && kw == 1 && sh == 1 && sw == 1 && ph == 0 && pw == 0 && kpad == cin && hl == H && wl == W);
  const i32 P = hl * wl, CHK = 64;
  i16 *wr0 = c->wrow, *wr1 = c->wrow + kpad;
  for (i32 p0 = 0; p0 < P; p0 += CHK) {
    const i32 rows = (P - p0 < CHK) ? P - p0 : CHK, rows4 = (rows + 3) & ~3;
    const i16 *A;
    if (direct) A = qin + (size_t)p0 * cin;
    else {
      i16 *col = c->col; A = col;
      for (i32 r = 0; r < rows4; r++) {
        i16 *d = col + (size_t)r * kpad;
        if (r >= rows) { for (i32 k = 0; k < kpad; k++) d[k] = 0; continue; }
        const i32 p = p0 + r, oy = p / wl, ox = p - oy * wl;
        i32 k = 0;
        for (i32 ky = 0; ky < kh; ky++) {
          const i32 iy = oy * sh - ph + ky;
          for (i32 kx = 0; kx < kw; kx++, k += cin) {
            const i32 ix = ox * sw - pw + kx;
            if ((unsigned)iy < (unsigned)H && (unsigned)ix < (unsigned)W) {
              const i16 *s = qin + ((size_t)iy * W + ix) * cin;
              for (i32 q = 0; q < cin; q++) d[k + q] = s[q];
            } else for (i32 q = 0; q < cin; q++) d[k + q] = 0;
          }
        }
        for (; k < kpad; k++) d[k] = 0;
      }
    }
    for (i32 co = 0; co < cout; co += 2) {
      const i32 two = co + 1 < cout;
      unpack_row(wq + (size_t)co * rowb, kpad, bits, wr0);
      if (two) unpack_row(wq + (size_t)(co + 1) * rowb, kpad, bits, wr1); else for (i32 k = 0; k < kpad; k++) wr1[k] = 0;
      const uint16_t *s0 = ws + (size_t)co * ng, *s1 = ws + (size_t)(co + 1) * ng;
      for (i32 r = 0; r < rows4; r += 4) {
        const i16 *a0 = A + (size_t)r * kpad, *a1 = a0 + kpad, *a2 = a1 + kpad, *a3 = a2 + kpad;
        v128_t f00 = wasm_i32x4_splat(0), f01 = f00, f10 = f00, f11 = f00, f20 = f00, f21 = f00, f30 = f00, f31 = f00;
        for (i32 g = 0; g < ng; g++) {
          v128_t i00 = wasm_i32x4_splat(0), i01 = i00, i10 = i00, i11 = i00, i20 = i00, i21 = i00, i30 = i00, i31 = i00;
          const i32 b = g << 5;
          for (i32 s = 0; s < 32; s += 8) {
            const v128_t w0 = wasm_v128_load(wr0 + b + s), w1 = wasm_v128_load(wr1 + b + s);
            v128_t a = wasm_v128_load(a0 + b + s); i00 = wasm_i32x4_add(i00, DOT(a, w0)); i01 = wasm_i32x4_add(i01, DOT(a, w1));
            a = wasm_v128_load(a1 + b + s); i10 = wasm_i32x4_add(i10, DOT(a, w0)); i11 = wasm_i32x4_add(i11, DOT(a, w1));
            a = wasm_v128_load(a2 + b + s); i20 = wasm_i32x4_add(i20, DOT(a, w0)); i21 = wasm_i32x4_add(i21, DOT(a, w1));
            a = wasm_v128_load(a3 + b + s); i30 = wasm_i32x4_add(i30, DOT(a, w0)); i31 = wasm_i32x4_add(i31, DOT(a, w1));
          }
          const v128_t c0 = wasm_f32x4_splat(bf16(s0[g])), c1 = wasm_f32x4_splat(two ? bf16(s1[g]) : 0.f);
#define ACC(f, i, cc) f = wasm_f32x4_add(f, wasm_f32x4_mul(wasm_f32x4_convert_i32x4(i), cc))
          ACC(f00, i00, c0); ACC(f01, i01, c1); ACC(f10, i10, c0); ACC(f11, i11, c1);
          ACC(f20, i20, c0); ACC(f21, i21, c1); ACC(f30, i30, c0); ACC(f31, i31, c1);
        }
        const v128_t fv[8] = { f00, f01, f10, f11, f20, f21, f30, f31 };
        for (i32 j = 0; j < 4; j++) {
          if (r + j >= rows) break;
          const i32 p = p0 + r + j, oy = p / wl, ox = p - oy * wl;
          const size_t pix = ((size_t)(oy * osy + ooy) * OW + (ox * osx + oox)) * cout;
          for (i32 k = 0; k < 1 + two; k++) {
            float v = hsum(fv[j * 2 + k]) * sa + (bias ? bias[co + k] : 0.f);
            if (res) v += res[pix + co + k];
            out[pix + co + k] = act_f(v, act);
          }
        }
      }
    }
  }
}

static void op_dw(const Model *md, Ctx *c, const i32 *o) {
  float *ar = c->arena;
  const i32 *ti = md->T + 5 * o[F_IN0], *to = md->T + 5 * o[F_OUT];
  const float *x = ar + ti[0]; float *out = ar + to[0];
  const i32 H = ti[2], W = ti[3], C = ti[4], kh = o[F_KH], kw = o[F_KW], sh = o[F_SH], sw = o[F_SW], ph = o[F_PH], pw = o[F_PW];
  const i32 OH = to[2], OW = to[3], act = o[F_ACT];
  const uint8_t *wq = md->blob + o[F_WOFF];
  const float *sc = (const float *)(md->blob + o[F_SOFF]), *bias = (const float *)(md->blob + o[F_BOFF]);
  float *wf = c->wf;
  for (i32 i = 0; i < kh * kw * C; i++) {
    uint8_t b = wq[i >> 1]; i32 q = (i & 1 ? b >> 4 : b & 15) - 8;
    wf[i] = (float)q * sc[i % C];
  }
  for (i32 oy = 0; oy < OH; oy++) for (i32 ox = 0; ox < OW; ox++) {
    float *op = out + ((size_t)oy * OW + ox) * C;
    for (i32 k = 0; k < C; k++) op[k] = bias[k];
    for (i32 ky = 0; ky < kh; ky++) {
      const i32 iy = oy * sh - ph + ky; if ((unsigned)iy >= (unsigned)H) continue;
      for (i32 kx = 0; kx < kw; kx++) {
        const i32 ix = ox * sw - pw + kx; if ((unsigned)ix >= (unsigned)W) continue;
        const float *ip = x + ((size_t)iy * W + ix) * C, *wp = wf + (size_t)(ky * kw + kx) * C;
        for (i32 k = 0; k < C; k++) op[k] += ip[k] * wp[k];
      }
    }
    if (act) for (i32 k = 0; k < C; k++) op[k] = act_f(op[k], act);
  }
}

static void run_net(i32 m, i32 t) {
  const Model *md = &M[m]; Ctx *c = &CX[m][t]; float *ar = c->arena;
  for (i32 i = 0; i < md->no; i++) {
    const i32 *o = md->O + (size_t)i * NF;
    const i32 *to = md->T + 5 * o[F_OUT];
    float *out = ar + to[0]; const i32 osz = to[1];
    switch (o[F_TYPE]) {
    case OP_CONV: op_conv(md, c, o); break;
    case OP_DW: op_dw(md, c, o); break;
    case OP_ADD: case OP_RELU: {
      const float *a = ar + md->T[5 * o[F_IN0]], *b = o[F_IN1] >= 0 ? ar + md->T[5 * o[F_IN1]] : 0;
      for (i32 k = 0; k < osz; k++) out[k] = act_f(a[k] + (b ? b[k] : 0.f), o[F_ACT]);
      break; }
    case OP_CONCAT: {
      const float *a = ar + md->T[5 * o[F_IN0]], *b = ar + md->T[5 * o[F_IN1]];
      const i32 O = o[F_X0], c0 = o[F_X1], c1 = o[F_X2];
      for (i32 r = 0; r < O; r++) {
        for (i32 k = 0; k < c0; k++) out[(size_t)r * (c0 + c1) + k] = a[(size_t)r * c0 + k];
        for (i32 k = 0; k < c1; k++) out[(size_t)r * (c0 + c1) + c0 + k] = b[(size_t)r * c1 + k];
      }
      break; }
    case OP_PAD: {
      const i32 *ti = md->T + 5 * o[F_IN0]; const float *a = ar + ti[0];
      const i32 pt = o[F_X0], pl = o[F_X1], pc = o[F_X2], H = ti[2], W = ti[3], C = ti[4], OW = to[3], OC = to[4];
      for (i32 k = 0; k < osz; k++) out[k] = 0.f;
      for (i32 y = 0; y < H; y++) for (i32 xx = 0; xx < W; xx++)
        for (i32 k = 0; k < C; k++) out[((size_t)(y + pt) * OW + (xx + pl)) * OC + pc + k] = a[((size_t)y * W + xx) * C + k];
      break; }
    case OP_POOL: {
      const i32 *ti = md->T + 5 * o[F_IN0]; const float *a = ar + ti[0];
      const i32 H = ti[2], W = ti[3], C = ti[4], OH = to[2], OW = to[3], kh = o[F_KH], kw = o[F_KW], sh = o[F_SH], sw = o[F_SW], ph = o[F_PH], pw = o[F_PW];
      for (i32 oy = 0; oy < OH; oy++) for (i32 ox = 0; ox < OW; ox++) for (i32 k = 0; k < C; k++) {
        float v = -3.4e38f;
        for (i32 ky = 0; ky < kh; ky++) { i32 iy = oy * sh - ph + ky; if ((unsigned)iy >= (unsigned)H) continue;
          for (i32 kx = 0; kx < kw; kx++) { i32 ix = ox * sw - pw + kx; if ((unsigned)ix >= (unsigned)W) continue;
            float u = a[((size_t)iy * W + ix) * C + k]; v = u > v ? u : v; } }
        out[((size_t)oy * OW + ox) * C + k] = v;
      }
      break; }
    }
  }
}

EXPORT("run_job") i32 run_job(i32 m, i32 t, i32 j) {
  const Model *md = &M[m]; float *ar = CX[m][t].arena;
  for (i32 i = 0; i < md->nin; i++) {
    const i32 *ts = md->T + 5 * md->IO[i]; const float *src = g_io[m][i] + (size_t)j * ts[1];
    for (i32 k = 0; k < ts[1]; k++) ar[ts[0] + k] = src[k];
  }
  run_net(m, t);
  for (i32 i = md->nin; i < md->nio; i++) {
    const i32 *ts = md->T + 5 * md->IO[i]; float *dst = g_io[m][i] + (size_t)j * ts[1];
    for (i32 k = 0; k < ts[1]; k++) dst[k] = ar[ts[0] + k];
  }
  return 0;
}
