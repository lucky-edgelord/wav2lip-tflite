"""End-to-end lip-sync on the W4 engine: BlazeFace (INT4) -> crop -> Wav2Lip (INT4) -> paste -> MP4."""
import subprocess, time, numpy as np, cv2

def _anchors():
    a = []
    for fm, n in ((16, 2), (8, 6)):
        for y in range(fm):
            for x in range(fm):
                a += [((x + .5) / fm, (y + .5) / fm)] * n
    return np.array(a, np.float32)
ANCH = _anchors()

def _mask():  # blend only the lower face: the upper half is the untouched reference, a hard box edge shows as a seam
    m = np.ones((96, 96), np.float32)
    m[:40] = 0; m[40:54] = np.linspace(0, 1, 14)[:, None]
    for k in range(7): m[:, k] *= (k + 1) / 8; m[:, 95 - k] *= (k + 1) / 8
    m[88:] *= np.linspace(1, 0.4, 8)[:, None]
    return m
MASK = _mask()

def decode(reg, sc, thr=0.5, iou_thr=0.3):
    """reg (896,16), sc (896,) raw -> (x1,y1,x2,y2 normalized, score) of the best face, or None."""
    s = 1 / (1 + np.exp(-np.clip(sc, -50, 50))); keep = np.where(s >= thr)[0]
    if not len(keep): return None
    cx = reg[keep, 0] / 128 + ANCH[keep, 0]; cy = reg[keep, 1] / 128 + ANCH[keep, 1]
    w = reg[keep, 2] / 128; h = reg[keep, 3] / 128
    bx = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1); sk = s[keep]; b = bx[np.argmax(sk)]
    iw = np.clip(np.minimum(bx[:, 2], b[2]) - np.maximum(bx[:, 0], b[0]), 0, None); ih = np.clip(np.minimum(bx[:, 3], b[3]) - np.maximum(bx[:, 1], b[1]), 0, None)
    inter = iw * ih; iou = inter / ((bx[:, 2] - bx[:, 0]) * (bx[:, 3] - bx[:, 1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)
    m = iou > iou_thr; wt = sk[m]
    return (bx[m] * wt[:, None]).sum(0) / wt.sum(), float(sk.max())

def detect(eng, frames_bgr):
    n = len(frames_bgr)
    x = np.stack([(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32) - 127.5) / 127.5 for f in frames_bgr])
    reg, sc = eng.run(1, [x], n)
    return [decode(reg[i * 896 * 16:(i + 1) * 896 * 16].reshape(896, 16), sc[i * 896:(i + 1) * 896]) for i in range(n)]

def lipsync(eng, video, audio, out, mel_chunks, melspectrogram, load_wav, max_width=640, pad_bottom=0.05, smooth=0.6,
            batch=16, max_frames=None, fps=None, log=print):
    wav = load_wav(audio); mel = melspectrogram(wav)
    cap = cv2.VideoCapture(video); vfps = cap.get(cv2.CAP_PROP_FPS) or 25; fps = fps or vfps
    frames = []
    while True:
        ok, f = cap.read()
        if not ok: break
        s = min(1, max_width / f.shape[1]); frames.append(cv2.resize(f, (int(f.shape[1] * s) // 2 * 2, int(f.shape[0] * s) // 2 * 2)) if s < 1 else f)
    if not frames: raise RuntimeError('could not read any frame from the video')
    chunks = mel_chunks(mel, fps); n = len(chunks) if not max_frames else min(len(chunks), max_frames)
    H, W = frames[0].shape[:2]; log(f'{len(frames)} source frames {W}x{H}, {n} output frames at {fps:g} fps, audio {len(wav)/16000:.1f}s')
    silent = out + '.silent.mp4'; vw = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    prev = None; t = dict(detect=0, wav2lip=0, other=0); found = 0; t0 = time.time()
    for i0 in range(0, n, batch):
        idx = list(range(i0, min(n, i0 + batch))); src = [frames[i % len(frames)].copy() for i in idx]
        s = time.time(); dets = detect(eng, src); t['detect'] += time.time() - s; s = time.time()
        boxes = []
        for f, d in zip(src, dets):
            if d is not None:
                found += 1; bb = d[0]; y2 = bb[3] + (bb[3] - bb[1]) * pad_bottom
                box = np.array([bb[0] * W, bb[1] * H, bb[2] * W, y2 * H])
                prev = box if prev is None else smooth * prev + (1 - smooth) * box
            boxes.append(None if prev is None else (max(0, int(prev[0])), max(0, int(prev[1])), min(W, int(prev[2])), min(H, int(prev[3])))
                         if int(prev[2]) - int(prev[0]) >= 8 and int(prev[3]) - int(prev[1]) >= 8 else None)
        melb = np.zeros((len(idx), 80, 16, 1), np.float32); faceb = np.zeros((len(idx), 96, 96, 6), np.float32)
        for k, (f, b, i) in enumerate(zip(src, boxes, idx)):
            melb[k, :, :, 0] = chunks[i]
            if b is None: continue
            c = cv2.resize(f[b[1]:b[3], b[0]:b[2]], (96, 96), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
            m = c.copy(); m[48:] = 0; faceb[k, :, :, :3] = m; faceb[k, :, :, 3:] = c
        t['other'] += time.time() - s; s = time.time()
        pred, = eng.run(0, [melb, faceb], len(idx)); t['wav2lip'] += time.time() - s; s = time.time()
        pred = pred.reshape(len(idx), 96, 96, 3)
        for f, b, p in zip(src, boxes, pred):
            if b is not None:
                sz = (b[2] - b[0], b[3] - b[1]); a = cv2.resize(MASK, sz, interpolation=cv2.INTER_LINEAR)[..., None]
                patch = cv2.resize((p * 255).clip(0, 255).astype(np.uint8), sz, interpolation=cv2.INTER_CUBIC).astype(np.float32)
                roi = f[b[1]:b[3], b[0]:b[2]]; roi[:] = (roi * (1 - a) + patch * a).astype(np.uint8)
            vw.write(f)
        t['other'] += time.time() - s
        log(f'frames {idx[-1]+1}/{n}  detect {1000*t["detect"]/(idx[-1]+1):.1f} ms/f  wav2lip {1000*t["wav2lip"]/(idx[-1]+1):.1f} ms/f')
    vw.release()
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', silent, '-i', audio, '-map', '0:v', '-map', '1:a', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                    '-c:a', 'aac', '-shortest', out], check=True)
    if not found: log('WARNING: no face detected in any frame; nothing was lip-synced')
    return dict(frames=n, faces=found, secs=time.time() - t0, **{k: v / n * 1000 for k, v in t.items()})
