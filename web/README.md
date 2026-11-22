# Web version

**Live page: [https://lucky-edgelord.github.io/wav2lip-tflite/](https://lucky-edgelord.github.io/wav2lip-tflite/)**

Wav2Lip lip-sync that runs entirely in the browser: pick a face video (or photo) and a driving
audio track, and it renders an MP4 locally. Nothing is uploaded.

The full-INT8 Wav2Lip model ships with the site (`assets/`, ~37 MB) and loads automatically. It runs
on the CPU through LiteRT.js (XNNPACK), multithreaded: up to 4 threads on phones, 8 on desktops.
On an Android phone (Adreno 6xx, 4 GB) that is ~330 ms/frame with 4 threads vs ~1060 ms single-threaded.

Multithreading needs cross-origin isolation. GitHub Pages cannot send the headers, so `coi-sw.js`
(a 20-line service worker) adds them; the page reloads itself once on the first visit.

**WebGPU was dropped.** LiteRT.js on WebGPU ran the model but its output ignored the audio (the
mouth never opened), on an Adreno 6xx phone and on SwiftShader, at fp16 and fp32 alike. The audio
encoder itself matched exactly on WebGPU, so the fault is later in the graph. GPU/NPU work moves to
the native Android app.

The model's input/output scales live in `assets/models.json` (LiteRT.js does not expose
quantization parameters). Rebuild with
`python scripts/export_web_models.py --checkpoint checkpoints/wav2lip_gan.pth --calib calib.npz`.
You can still load your own fp32/int8w model under Settings.

## Log console

The **Log** button (bottom right) opens a built-in console: every step with timings, an environment
report (browser, device, WebGPU adapter, WebCodecs, storage, connection), download progress and
retries, per-batch timings in **Verbose**, console output from LiteRT.js/MediaPipe, and full error
details with stack traces. It opens by itself on an error. **Copy** or **Save .txt** to share a log.

Failure handling: model download retries (3×, 15 s stall watchdog) and is cached on the device;
picked files are read into memory immediately; audio falls back from Web Audio to WebCodecs;
the face detector falls back from GPU to CPU and restarts if it fails mid-run; the screen is kept
awake while rendering.

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
