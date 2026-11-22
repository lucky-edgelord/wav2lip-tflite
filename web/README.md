# Web version

**Live page: [https://lucky-edgelord.github.io/wav2lip-tflite/](https://lucky-edgelord.github.io/wav2lip-tflite/)**

Wav2Lip lip-sync that runs entirely in the browser: pick a face video (or photo) and a driving
audio track, and it renders an MP4 locally. Nothing is uploaded.

The INT8 Wav2Lip models ship with the site (`assets/`, ~37 MB each) and load automatically:

| device | model | why |
|---|---|---|
| WebGPU available | `wav2lip_gan_b8_int8w.tflite`: INT8 weights, float I/O | WebGPU expands weights to float; 49 dB vs PyTorch on an unseen face |
| CPU only | `wav2lip_gan_b8_int8.tflite`: full INT8, int8 I/O | 3.4× faster than FP32 in Wasm; 35 dB on an unseen face (faint grain) |

The full-INT8 model's input/output scales live in `assets/models.json` because LiteRT.js does not
expose quantization parameters. Rebuild both with
`python scripts/export_web_models.py --checkpoint checkpoints/wav2lip_gan.pth --calib calib.npz`
(calibration uses `--augment-calib` so brighter faces and louder audio do not clip).
You can still load your own fp32/int8w model under Settings.

- **Wav2Lip**: [LiteRT.js](https://www.npmjs.com/package/@litertjs/core) on WebGPU, with a CPU (Wasm) fallback
- **Face boxes**: MediaPipe Tasks Vision `FaceDetector` (BlazeFace short-range), same padding/smoothing as `w2l/face.py`
- **Audio features**: `src/mel.js`, a port of `w2l/audio.py` (matches librosa to 2e-5)
- **Video in/out**: [Mediabunny](https://mediabunny.dev) on WebCodecs (H.264 + AAC/Opus MP4)

## Log console

The **Log** button (bottom right) opens a built-in console: every step with timings, an environment
report (browser, device, WebGPU adapter, WebCodecs, storage, connection), download progress and
retries, per-batch timings in **Verbose**, console output from LiteRT.js/MediaPipe, and full error
details with stack traces. It opens by itself on an error. **Copy** or **Save .txt** to share a log.

Failure handling: model download retries (3×, 15 s stall watchdog) and is cached on the device;
picked files are read into memory immediately; audio falls back from Web Audio to WebCodecs;
the face detector falls back from GPU to CPU and restarts if it fails mid-run; Wav2Lip falls back
from WebGPU to CPU at load or mid-run; the screen is kept awake while rendering.

## Run locally

```bash
cd web
npm install
npm run build          # writes dist/ (page, bundle, Wasm, BlazeFace model)
npx serve dist         # or: python3 -m http.server -d dist
```

Open the printed URL in Chrome or Edge (WebGPU + WebCodecs; Safari 26+ should also work).

## Deploy to GitHub Pages

Settings → Pages → Source: **GitHub Actions**. The **Deploy web** workflow runs on every push that
touches `web/` (or run it by hand from the Actions tab).

## Tested

Headless Chrome 153 on Linux with no GPU (WebGPU via SwiftShader), real `wav2lip_gan` weights,
640×640 clip + 8.5 s of speech:

| backend | result | speed (software-only machine) |
|---|---|---|
| CPU, FP32 | 209 frames, H.264 + Opus MP4, frames match the Python pipeline | 3.1 fps (245 ms/frame Wav2Lip) |
| CPU, full INT8 (built-in) | correct lip shapes, faint grain | 6.4 fps (72 ms/frame Wav2Lip) |
| WebGPU, FP32 / INT8 weights (SwiftShader) | correct output | 0.1–0.4 fps (software GPU emulation; not representative) |

Real GPU numbers (e.g. M1 / Metal via WebGPU) are still to be measured.
