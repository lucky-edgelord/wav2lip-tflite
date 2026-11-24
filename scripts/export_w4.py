"""Export Wav2Lip and BlazeFace to the W4G1 format read by w4/w4.c.

Weights: symmetric INT4 in groups of 32 along the reduction axis, one bf16 scale per group, with a
per-group clipping search. Everything else (activations, bias, residuals) stays float and the engine
quantizes activations to 8 bits per tensor on the fly.

  python scripts/export_w4.py wav2lip --ckpt checkpoints/wav2lip_gan.pth --out web/assets/w2l_w4.bin
  python scripts/export_w4.py blaze --tflite web/models/blaze_face_short_range.tflite --out web/assets/blaze_w4.bin
"""
import argparse, struct, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
NF = 40
(F_TYPE, F_IN0, F_IN1, F_OUT, F_ACT, F_RES, F_CIN, F_COUT, F_KH, F_KW, F_SH, F_SW, F_PH, F_PW, F_HL, F_WL,
 F_OSY, F_OSX, F_OOY, F_OOX, F_KPAD, F_WBITS, F_WOFF, F_SOFF, F_BOFF, F_X0, F_X1, F_X2, F_X3) = range(29)
OP_CONV, OP_DW, OP_ADD, OP_CONCAT, OP_PAD, OP_POOL, OP_RELU = range(7)
ACT_NONE, ACT_RELU, ACT_SIGMOID = 0, 1, 2
CHUNK = 64


def to_bf16(x):
    """Round float32 to bfloat16 (returns uint16 codes and the rounded float32 values)."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return r, (r.astype(np.uint32) << 16).view(np.float32)


def quant_groups(W2, bits):
    """W2: (rows, K) float. Returns (packed bytes, bf16 scales (rows, K/32), dequantized W2)."""
    rows, K = W2.shape
    kp = (K + 31) // 32 * 32
    Wp = np.zeros((rows, kp), np.float32); Wp[:, :K] = W2
    g = Wp.reshape(rows, kp // 32, 32)
    amax = np.abs(g).max(-1, keepdims=True) + 1e-12
    qmax = 7 if bits == 4 else 127
    qlo = -8 if bits == 4 else -128
    best_e = np.full(amax.shape, np.inf, np.float32); best_s = np.zeros_like(amax); best_q = np.zeros_like(g)
    for f in np.linspace(0.55, 1.0, 12) if bits == 4 else [1.0]:
        _, s = to_bf16(amax * f / qmax)
        q = np.clip(np.round(g / s), qlo, qmax)
        e = ((q * s - g) ** 2).sum(-1, keepdims=True)
        better = e < best_e
        best_e = np.where(better, e, best_e); best_s = np.where(better, s, best_s); best_q = np.where(better, q, best_q)
    codes, sc = to_bf16(best_s[..., 0])
    deq = (best_q * best_s).reshape(rows, kp)
    if bits == 4:
        u = (best_q + 8).astype(np.uint8)                       # (rows, ng, 32)
        packed = (u[..., :16] | (u[..., 16:] << 4)).reshape(rows, -1)
    else:
        packed = best_q.astype(np.int8).view(np.uint8).reshape(rows, -1)
    return packed.tobytes(), codes, deq[:, :K]


class Graph:
    def __init__(s):
        s.t, s.ops, s.blob, s.inputs, s.outputs = [], [], bytearray(), [], []
        s.max_qin = s.max_col = s.max_wrow = 0; s.abits = 8
    # --- tensors
    def tensor(s, h, w, c, base=None):
        s.t.append(dict(h=h, w=w, c=c, lo=10**9, hi=-1, base=base)); return len(s.t) - 1
    def root(s, i):
        while s.t[i]['base'] is not None: i = s.t[i]['base']
        return i
    def alias(s, i, h, w, c): return s.tensor(h, w, c, base=s.root(i))
    def use(s, i, k): r = s.root(i); s.t[r]['hi'] = max(s.t[r]['hi'], k); s.t[r]['lo'] = min(s.t[r]['lo'], k)
    def define(s, i, k): r = s.root(i); s.t[r]['lo'] = min(s.t[r]['lo'], k); s.t[r]['hi'] = max(s.t[r]['hi'], k)
    def size(s, i): t = s.t[i]; return t['h'] * t['w'] * t['c']
    def input(s, h, w, c):
        i = s.tensor(h, w, c); s.inputs.append(i); s.define(i, -1); return i
    # --- blob
    def add_blob(s, b):
        while len(s.blob) % 64: s.blob.append(0)
        off = len(s.blob); s.blob += bytes(b); return off
    # --- ops
    def emit(s, typ, ins, out, **kw):
        k = len(s.ops); f = [0] * NF
        f[F_TYPE] = typ; f[F_IN0] = ins[0] if ins else -1; f[F_IN1] = ins[1] if len(ins) > 1 else -1; f[F_OUT] = out
        f[F_RES] = -1; f[F_BOFF] = -1; f[F_OSY] = f[F_OSX] = 1
        names = dict(act=F_ACT, res=F_RES, cin=F_CIN, cout=F_COUT, kh=F_KH, kw=F_KW, sh=F_SH, sw=F_SW, ph=F_PH, pw=F_PW,
                     hl=F_HL, wl=F_WL, osy=F_OSY, osx=F_OSX, ooy=F_OOY, oox=F_OOX, kpad=F_KPAD, wbits=F_WBITS, woff=F_WOFF,
                     soff=F_SOFF, boff=F_BOFF, x0=F_X0, x1=F_X1, x2=F_X2)
        for a, v in kw.items(): f[names[a]] = v
        for i in ins: s.use(i, k)
        if f[F_RES] >= 0: s.use(f[F_RES], k)
        s.define(out, k); s.ops.append(f); return out
    def conv(s, x, W, b, stride=(1, 1), pad=(0, 0), act=ACT_NONE, res=None, bits=4, out=None, hl=None, wl=None,
             scatter=(1, 1, 0, 0), out_hw=None):
        """W: (cout, kh, kw, cin). Output grid hl x wl; output pixel (oy*osy+ooy, ox*osx+oox)."""
        cout, kh, kw, cin = W.shape; t = s.t[x]
        hl = hl or (t['h'] + 2 * pad[0] - kh) // stride[0] + 1
        wl = wl or (t['w'] + 2 * pad[1] - kw) // stride[1] + 1
        if out is None: out = s.tensor(hl, wl, cout)
        K = kh * kw * cin; kpad = (K + 31) // 32 * 32
        packed, codes, _ = quant_groups(W.reshape(cout, K).astype(np.float32), bits)
        woff = s.add_blob(packed); soff = s.add_blob(codes.astype('<u2').tobytes())
        boff = s.add_blob(np.asarray(b, '<f4').tobytes()) if b is not None else -1
        P = hl * wl; direct = (kh == kw == 1 and stride == (1, 1) and pad == (0, 0) and kpad == cin and hl == t['h'] and wl == t['w'])
        s.max_qin = max(s.max_qin, s.size(x))
        if not direct: s.max_col = max(s.max_col, ((min(CHUNK, P) + 3) // 4 * 4) * kpad)
        s.max_wrow = max(s.max_wrow, kpad)
        return s.emit(OP_CONV, [x], out, act=act, res=res if res is not None else -1, cin=cin, cout=cout, kh=kh, kw=kw,
                      sh=stride[0], sw=stride[1], ph=pad[0], pw=pad[1], hl=hl, wl=wl, osy=scatter[0], osx=scatter[1],
                      ooy=scatter[2], oox=scatter[3], kpad=kpad, wbits=bits, woff=woff, soff=soff, boff=boff)
    def dw(s, x, W, b, stride, pad, out_hw, act=ACT_NONE, bits=4):
        """W: (kh, kw, C). Per-channel scale over the kh*kw taps."""
        kh, kw, C = W.shape; Wc = W.reshape(kh * kw, C)
        amax = np.abs(Wc).max(0) + 1e-12; best = None
        for f in np.linspace(0.6, 1.0, 9):
            sc = amax * f / 7; q = np.clip(np.round(Wc / sc), -8, 7); e = ((q * sc - Wc) ** 2).sum(0)
            if best is None: best = (e, sc, q)
            else:
                m = e < best[0]; best = (np.where(m, e, best[0]), np.where(m, sc, best[1]), np.where(m[None], q, best[2]))
        sc, q = best[1].astype('<f4'), best[2]
        u = (q + 8).astype(np.uint8).reshape(-1)
        if len(u) & 1: u = np.append(u, 8).astype(np.uint8)
        packed = (u[0::2] | (u[1::2] << 4)).astype(np.uint8)
        out = s.tensor(out_hw[0], out_hw[1], C)
        s.max_wrow = max(s.max_wrow, kh * kw * C)
        return s.emit(OP_DW, [x], out, act=act, cin=C, cout=C, kh=kh, kw=kw, sh=stride[0], sw=stride[1], ph=pad[0], pw=pad[1],
                      woff=s.add_blob(packed), soff=s.add_blob(sc.tobytes()), boff=s.add_blob(np.asarray(b, '<f4').tobytes()))
    def add(s, a, b, act=ACT_NONE): return s.emit(OP_ADD, [a, b], s.tensor(s.t[a]['h'], s.t[a]['w'], s.t[a]['c']), act=act)
    def relu(s, a): return s.emit(OP_RELU, [a], s.tensor(s.t[a]['h'], s.t[a]['w'], s.t[a]['c']), act=ACT_RELU)
    def concat_c(s, a, b):
        ta, tb = s.t[a], s.t[b]
        return s.emit(OP_CONCAT, [a, b], s.tensor(ta['h'], ta['w'], ta['c'] + tb['c']), x0=ta['h'] * ta['w'], x1=ta['c'], x2=tb['c'])
    def concat_flat(s, a, b, outer, ca, cb):  # a, b already shaped so that outer*ca / outer*cb elements
        return s.emit(OP_CONCAT, [a, b], s.tensor(1, outer, ca + cb), x0=outer, x1=ca, x2=cb)
    def pad(s, a, pt, pl, pc, oh, ow, oc): return s.emit(OP_PAD, [a], s.tensor(oh, ow, oc), x0=pt, x1=pl, x2=pc)
    def pool(s, a, k, stride, pad, oh, ow):
        return s.emit(OP_POOL, [a], s.tensor(oh, ow, s.t[a]['c']), kh=k[0], kw=k[1], sh=stride[0], sw=stride[1], ph=pad[0], pw=pad[1])
    def mark_output(s, i): s.outputs.append(i); s.use(i, len(s.ops))

    def allocate(s):
        order = sorted([i for i in range(len(s.t)) if s.t[i]['base'] is None and s.t[i]['hi'] >= 0], key=lambda i: (s.t[i]['lo'], -s.size(i)))
        placed = []; top = 0
        for i in order:
            t = s.t[i]; sz = (s.size(i) + 15) // 16 * 16
            busy = sorted((o, o + z) for (o, z, lo, hi) in placed if not (hi < t['lo'] or lo > t['hi']))
            off = 0
            for a, b in busy:
                if off + sz <= a: break
                off = max(off, b)
            t['off'] = off; placed.append((off, sz, t['lo'], t['hi'])); top = max(top, off + sz)
        for i, t in enumerate(s.t):
            if t['base'] is not None: t['off'] = s.t[s.root(i)]['off']
        return top

    def write(s, path):
        arena = s.allocate()
        io = s.inputs + s.outputs
        tens = b''.join(struct.pack('<5i', t.get('off', 0), t['h'] * t['w'] * t['c'], t['h'], t['w'], t['c']) for t in s.t)
        ops = b''.join(struct.pack(f'<{NF}i', *f) for f in s.ops)
        iob = struct.pack(f'<{len(io)}i', *io)
        pos = 64
        def al(n): return (n + 63) // 64 * 64
        o_t = pos; pos = al(pos + len(tens)); o_o = pos; pos = al(pos + len(ops)); o_io = pos; pos = al(pos + len(iob)); o_b = pos
        hdr = struct.pack('<16I', 0x31473457, len(s.t), len(s.ops), arena, s.max_qin, s.max_col, s.max_wrow, len(io), len(s.inputs),
                          o_t, o_o, o_io, o_b, s.abits, 0, 0)
        buf = bytearray(hdr.ljust(64, b'\0'))
        for off, data in ((o_t, tens), (o_o, ops), (o_io, iob), (o_b, bytes(s.blob))):
            buf.extend(b'\0' * (off - len(buf))); buf.extend(data)
        Path(path).write_bytes(bytes(buf))
        print(f'{path}: {len(buf)/1e6:.2f} MB, {len(s.ops)} ops, arena {arena*4/1e6:.1f} MB/thread, qin {s.max_qin*2/1e6:.1f} MB, col {s.max_col*2/1e6:.2f} MB')


# ------------------------------------------------------------------ Wav2Lip
def build_wav2lip(ckpt, bits=4, edge_bits=4, abits=8):
    import torch
    from torch import nn
    from w2l.model import load_wav2lip, Conv2d, Conv2dTranspose
    model = load_wav2lip(ckpt)
    g = Graph(); g.abits = abits
    mel = g.input(80, 16, 1); face = g.input(96, 96, 6)

    def fold(mod):
        conv, bn = mod.conv_block
        sc = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach()
        w = conv.weight.detach(); b = conv.bias.detach() if conv.bias is not None else torch.zeros(bn.running_mean.shape)
        if isinstance(conv, nn.ConvTranspose2d): w = w * sc.view(1, -1, 1, 1)
        else: w = w * sc.view(-1, 1, 1, 1)
        return conv, w.numpy().astype(np.float32), ((b - bn.running_mean.detach()) * sc + bn.bias.detach()).numpy().astype(np.float32)

    def run(x, mod, first_last=False):
        conv, w, b = fold(mod)
        bt = edge_bits if first_last else bits
        if isinstance(conv, nn.ConvTranspose2d):
            assert conv.kernel_size == (3, 3)
            if conv.stride == (1, 1):          # 1x1 -> 3x3: one 1x1 conv producing the 9 positions
                assert g.t[x]['h'] == 1 and g.t[x]['w'] == 1 and conv.padding == (0, 0)
                cin, cout = w.shape[:2]
                Wm = w.transpose(2, 3, 1, 0).reshape(9 * cout, 1, 1, cin)      # row = (ky*3+kx)*cout + co
                out = g.tensor(3, 3, cout)
                g.conv(x, Wm, np.tile(b, 9), act=ACT_RELU, bits=bt, out=out, hl=1, wl=1)
                return out
            assert conv.stride == (2, 2) and conv.padding == (1, 1) and conv.output_padding == (1, 1)
            H, Wd, cin, cout = g.t[x]['h'], g.t[x]['w'], w.shape[0], w.shape[1]
            out = g.tensor(2 * H, 2 * Wd, cout)
            taps = {0: [1], 1: [2, 0]}                                       # kernel index per input offset dy=0,1
            for py in (0, 1):
                for px in (0, 1):
                    Wp = np.stack([np.stack([w[:, :, ky, kx].T for kx in taps[px]], 0) for ky in taps[py]], 0)  # (kh,kw,cout,cin)
                    Wp = Wp.transpose(2, 0, 1, 3)
                    g.conv(x, Wp, b, act=ACT_RELU, bits=bt, out=out, hl=H, wl=Wd, scatter=(2, 2, py, px))
            return out
        Wn = w.transpose(0, 2, 3, 1)
        return g.conv(x, Wn, b, stride=conv.stride, pad=conv.padding, act=ACT_RELU, res=x if mod.residual else None, bits=bt)

    feats = []; f = face; first = True
    for blk in model.face_encoder_blocks:
        for mod in blk:
            f = run(f, mod, first_last=first); first = False
        feats.append(f)
    x = mel; first = True
    for mod in model.audio_encoder:
        x = run(x, mod, first_last=first); first = False
    for blk in model.face_decoder_blocks:
        for mod in blk: x = run(x, mod)
        x = g.concat_c(x, feats.pop())
    x = run(x, model.output_block[0], first_last=True)
    c = model.output_block[1]
    out = g.conv(x, c.weight.detach().numpy().transpose(0, 2, 3, 1).astype(np.float32), c.bias.detach().numpy(), act=ACT_SIGMOID, bits=edge_bits)
    g.mark_output(out)
    return g


# ------------------------------------------------------------------ BlazeFace
def build_blaze(path, bits=4, dw_bits=4, abits=8):
    import tflite
    buf = Path(path).read_bytes()
    m = tflite.Model.GetRootAsModel(buf, 0); sg = m.Subgraphs(0)
    names = {v: k for k, v in vars(tflite.BuiltinOperator).items() if not k.startswith('_')}
    def const(ti):
        t = sg.Tensors(ti); b = m.Buffers(t.Buffer()).DataAsNumpy()
        if isinstance(b, int) or b.size == 0: return None
        dt = {0: np.float32, 1: np.float16, 2: np.int32}[t.Type()]
        return np.frombuffer(bytes(b), dt).reshape([int(v) for v in t.ShapeAsNumpy()]).copy()
    consts = {}
    g = Graph(); g.abits = abits; mp = {}
    def shape(ti):
        s = [int(v) for v in sg.Tensors(ti).ShapeAsNumpy()]
        if len(s) == 4: return s[1], s[2], s[3]
        if len(s) == 3: return 1, s[1], s[2]
        return 1, 1, int(np.prod(s))
    inp = sg.Inputs(0); mp[inp] = g.input(*shape(inp))
    def same_pad(i, o, k, s): return max(((o - 1) * s + k - i) // 2, 0)
    for oi in range(sg.OperatorsLength()):
        op = sg.Operators(oi); code = m.OperatorCodes(op.OpcodeIndex()); name = names[max(code.BuiltinCode(), code.DeprecatedBuiltinCode())]
        ins = [op.Inputs(j) for j in range(op.InputsLength())]; outs = [op.Outputs(j) for j in range(op.OutputsLength())]
        def opts(cls):
            o = cls(); o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos); return o
        if name == 'DEQUANTIZE':
            consts[outs[0]] = const(ins[0]).astype(np.float32); continue
        def cval(ti): return consts[ti] if ti in consts else const(ti).astype(np.float32)
        if name in ('CONV_2D', 'DEPTHWISE_CONV_2D'):
            o = opts(tflite.Conv2DOptions if name == 'CONV_2D' else tflite.DepthwiseConv2DOptions)
            x = mp[ins[0]]; H, Wd, _ = shape(ins[0]); oh, ow, _ = shape(outs[0])
            s = (o.StrideH(), o.StrideW()); w = cval(ins[1]); b = cval(ins[2]); act = ACT_RELU if o.FusedActivationFunction() == 1 else ACT_NONE
            assert o.FusedActivationFunction() in (0, 1)
            if name == 'CONV_2D':
                kh, kw = w.shape[1:3]
                pad = (same_pad(H, oh, kh, s[0]), same_pad(Wd, ow, kw, s[1])) if o.Padding() == 0 else (0, 0)
                mp[outs[0]] = g.conv(x, w, b, stride=s, pad=pad, act=act, bits=bits, hl=oh, wl=ow)
            else:
                assert o.DepthMultiplier() == 1
                kh, kw = w.shape[1:3]
                pad = (same_pad(H, oh, kh, s[0]), same_pad(Wd, ow, kw, s[1])) if o.Padding() == 0 else (0, 0)
                mp[outs[0]] = g.dw(x, w[0], b, s, pad, (oh, ow), act=act, bits=dw_bits)
        elif name == 'ADD':
            mp[outs[0]] = g.add(mp[ins[0]], mp[ins[1]], act=ACT_NONE)
        elif name == 'RELU': mp[outs[0]] = g.relu(mp[ins[0]])
        elif name == 'MAX_POOL_2D':
            o = opts(tflite.Pool2DOptions); H, Wd, _ = shape(ins[0]); oh, ow, _ = shape(outs[0])
            k = (o.FilterHeight(), o.FilterWidth()); s = (o.StrideH(), o.StrideW())
            pad = (same_pad(H, oh, k[0], s[0]), same_pad(Wd, ow, k[1], s[1])) if o.Padding() == 0 else (0, 0)
            mp[outs[0]] = g.pool(mp[ins[0]], k, s, pad, oh, ow)
        elif name == 'PAD':
            p = const(ins[1]); oh, ow, oc = shape(outs[0])
            assert p[0].sum() == 0
            mp[outs[0]] = g.pad(mp[ins[0]], int(p[1][0]), int(p[2][0]), int(p[3][0]), oh, ow, oc)
        elif name == 'RESHAPE':
            h, w, c = shape(outs[0]); mp[outs[0]] = g.alias(mp[ins[0]], h, w, c)
        elif name == 'CONCATENATION':
            o = opts(tflite.ConcatenationOptions); ax = o.Axis(); shp = [[int(v) for v in sg.Tensors(i).ShapeAsNumpy()] for i in ins]
            assert len(ins) == 2
            outer = int(np.prod(shp[0][:ax])); ca = int(np.prod(shp[0][ax:])); cb = int(np.prod(shp[1][ax:]))
            mp[outs[0]] = g.concat_flat(mp[ins[0]], mp[ins[1]], outer, ca, cb) if outer > 1 or True else None
        else: raise SystemExit(f'unsupported op {name}')
    outs = [sg.Outputs(i) for i in range(sg.OutputsLength())]
    for ti in outs: print('blaze output', ti, [int(v) for v in sg.Tensors(ti).ShapeAsNumpy()])
    for ti in sorted(outs, key=lambda t: -int(np.prod(sg.Tensors(t).ShapeAsNumpy()))): g.mark_output(mp[ti])  # regressors first
    return g


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('which', choices=['wav2lip', 'blaze'])
    ap.add_argument('--ckpt'); ap.add_argument('--tflite'); ap.add_argument('--out', required=True)
    ap.add_argument('--bits', type=int, default=4); ap.add_argument('--edge-bits', type=int, default=4); ap.add_argument('--abits', type=int, default=8)
    ap.add_argument('--dw-bits', type=int, default=4)
    a = ap.parse_args()
    g = build_wav2lip(a.ckpt, a.bits, a.edge_bits, a.abits) if a.which == 'wav2lip' else build_blaze(a.tflite, a.bits, a.dw_bits, a.abits)
    g.write(a.out)
