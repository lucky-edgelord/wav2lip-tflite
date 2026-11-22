# Web version

**Live page: [https://lucky-edgelord.github.io/wav2lip-tflite/](https://lucky-edgelord.github.io/wav2lip-tflite/)**

Wav2Lip lip-sync that runs entirely in the browser: pick a face video (or photo), a driving
audio track and a converted `.tflite` model, and it renders an MP4 locally. Nothing is uploaded.

- **Wav2Lip**: [LiteRT.js](https://www.npmjs.com/package/@litertjs/core) on WebGPU, with a CPU (Wasm) fallback
- **Face boxes**: MediaPipe Tasks Vision `FaceDetector` (BlazeFace short-range), same padding/smoothing as `w2l/face.py`
- **Audio features**: `src/mel.js`, a port of `w2l/audio.py` (matches librosa to 2e-5)
- **Video in/out**: [Mediabunny](https://mediabunny.dev) on WebCodecs (H.264 + AAC/Opus MP4)

## Run locally

```bash
cd web
npm install
npm run build          # writes dist/ (page, bundle, Wasm, BlazeFace model)
npx serve dist         # or: python3 -m http.server -d dist
```

Open the printed URL in Chrome or Edge (WebGPU + WebCodecs; Safari 26+ should also work).
Make the model with `python scripts/convert.py --checkpoint checkpoints/wav2lip_gan.pth --batch 1`
(fp32 for WebGPU; `int8w` is smaller but not faster; full `int8` has integer I/O and is not supported here).
The page can remember the model in IndexedDB so you only pick it once.

## Deploy to GitHub Pages

Settings → Pages → Source: **GitHub Actions**, then run **Deploy web** from the Actions tab.

## Tested

Headless Chrome 153 on Linux with no GPU (WebGPU via SwiftShader), real `wav2lip_gan` weights,
640×640 clip + 8.5 s of speech:

| backend | result | speed (software-only machine) |
|---|---|---|
| CPU (Wasm, XNNPACK) | 209 frames, H.264 + Opus MP4, frames match the Python pipeline | 3.1 fps (245 ms/frame Wav2Lip) |
| WebGPU (SwiftShader) | correct output | 0.4 fps (software GPU emulation; not representative) |

Real GPU numbers (e.g. M1 / Metal via WebGPU) are still to be measured.
