# Benchmarks

All numbers below were measured on **2026-11-22** in a Linux x86_64 cloud sandbox with
**no GPU** (LiteRT ran on CPU via XNNPACK; the GPU accelerator was requested and fell back).
`ai-edge-litert` 2.3.0, `litert-torch` 0.10.0, `ai-edge-quantizer` 0.10.1, real `wav2lip_gan.pth` weights.
Apple Silicon / Metal numbers are **not measured yet**; see "How to reproduce" to add them.

## Model only (`scripts/compare.py`)

Batch 8 models, inputs = 209 real face crops + mel windows dumped by `infer.py --dump-calib`,
error measured against the original PyTorch model on the same inputs.

| model | size | ms / frame (CPU) | max abs error | PSNR vs PyTorch |
|---|---:|---:|---:|---:|
| `fp32` | 145.2 MB | 90.4 | 0.0000 | 138.9 dB (bit-exact for practical purposes) |
| `int8w` (INT8 weights, float activations) | 36.6 MB | 93.8 | 0.0136 | 55.4 dB |
| `int8` (full INT8, calibrated on the same clip) | 36.7 MB | **17.7** | 0.1009 | 46.0 dB |

- Full INT8 is ~5× faster than FP32 on CPU. 46 dB is visually indistinguishable on the 96×96 crop
  (checked frame by frame on the test clip).
- `int8w` only shrinks the file: activations stay float, so CPU speed is unchanged. On GPU the
  weights are expanded back to float, so expect the same there.
- Batch 1 (random weights, same machine): FP32 118–146 ms/frame, full INT8 27–39 ms/frame.
  Batching helps CPU throughput by ~25%.

## End to end (`infer.py`)

640×640 @ 25 fps face video, 8.5 s of speech → 209 output frames, batch 8.

| model | total | throughput | face detect | Wav2Lip | everything else* |
|---|---:|---:|---:|---:|---:|
| `fp32` | 22.5 s | 9.3 fps | 5.1 ms/frame | 97.1 ms/frame | 5.3 ms/frame |
| `int8` | 6.5 s | **32.2 fps** | 4.8 ms/frame | 21.1 ms/frame | 5.1 ms/frame |

\* decode, crop/mask/resize, paste-back and MJPEG write; final ffmpeg mux not included.

What this says about where to optimise next:
- With FP32, Wav2Lip is ~90% of frame time, so only a faster model path (GPU, INT8) matters.
- With INT8, detection + glue is ~10 ms of ~31 ms per frame (about a third). That is the point where
  moving the CPU-side work to Rust, or keeping frames on the GPU with custom pre/post-processing
  kernels, starts to pay off.
- MediaPipe BlazeFace on CPU costs ~5 ms/frame at 640×640; it is not the bottleneck.

## Browser (`web/`, LiteRT.js)

Same clip, batch 8 FP32 model, headless Chrome 153 on the same CPU-only machine:

| backend | throughput | Wav2Lip | detect | other |
|---|---:|---:|---:|---:|
| CPU (Wasm), FP32 | 3.1 fps | 245 ms/frame | 68 ms/frame* | 8 ms/frame |
| CPU (Wasm), full INT8 | 6.4 fps | **72 ms/frame** | 73 ms/frame* | 10 ms/frame |
| WebGPU via SwiftShader (software), FP32 | 0.4 fps | 2429 ms/frame | 228 ms/frame* | 11 ms/frame |
| WebGPU via SwiftShader (software), INT8 weights | 0.1 fps | 6959 ms/frame | 242 ms/frame* | 13 ms/frame |

\* MediaPipe ran its "GPU" delegate on software WebGL here; on real hardware detection is a few ms.
Wasm is ~2.7× slower than native XNNPACK on the same machine. These are floor numbers; measure on a real GPU.

## INT8 on an unseen face

Calibrating on one clip fits the INT8 input ranges to it (pixels clipped at 0.88, audio at 0).
With `--augment-calib` (brightness and loudness variants) the ranges cover pixels 0–0.99 and audio
−3.95 to 3.05. Tested on a different face (astronaut portrait + the same speech), batch 8, CPU:

| model | max abs error | PSNR vs PyTorch | ms / frame |
|---|---:|---:|---:|
| `int8w` | 0.080 | 49.1 dB | 92.8 |
| `int8`, calibrated on one clip | 0.374 | 35.3 dB | 22.1 |
| `int8`, augmented calibration | 0.151 | 34.5 dB | 16.7 |

Full INT8 shows faint grain on unseen faces; `int8w` is visually identical to FP32.

## Conversion facts

- Converted graph ops: `CONV_2D` 45, `ADD` 28, `SLICE` 10, `PAD` 9, `CONCATENATION` 7,
  `TRANSPOSE_CONV` 5, `DEPTHWISE_CONV_2D` 1, `LOGISTIC` 1. No `TRANSPOSE` ops thanks to channels-last I/O.
- The cropped-transpose-conv rewrite matches the original layer to 6e-8; the converted FP32 model
  matches PyTorch to ~1e-7.
- Conversion time: ~10 s for FP32, ~1 min more for calibrated INT8.

## How to reproduce / add your machine

```bash
python infer.py --face face.mp4 --audio speech.wav --model models/wav2lip_gan_b8_fp32.tflite \
    --dump-calib calib.npz                       # also prints the end-to-end time split
python scripts/convert.py --checkpoint checkpoints/wav2lip_gan.pth --batch 8 \
    --precision fp32 int8w int8 --calib calib.npz
python scripts/compare.py --checkpoint checkpoints/wav2lip_gan.pth --calib calib.npz \
    models/wav2lip_gan_b8_*.tflite --accelerator gpu   # or cpu
```

Please add a row with your hardware (e.g. M1 Mac mini, Metal) when you run it.
