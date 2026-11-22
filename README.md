# wav2lip-tflite

Wav2Lip lip-sync running on [LiteRT](https://ai.google.dev/edge/litert) with GPU acceleration, using MediaPipe's BlazeFace face detector instead of S3FD. Built and tested for Apple Silicon (Metal), but runs anywhere `ai-edge-litert` does.

```
frame ─► MediaPipe BlazeFace ─► pad + smooth box ─► 96×96 crop ─► Wav2Lip (LiteRT GPU) ─► paste back
audio ─► mel spectrogram (80×16 per frame) ───────────────────────┘
```

## Setup (macOS arm64, Linux x86_64)

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # inference
pip install -r requirements-convert.txt    # conversion (adds torch + litert-torch)
brew install ffmpeg                        # or apt install ffmpeg
```

Get the original Wav2Lip checkpoints (`wav2lip.pth` / `wav2lip_gan.pth`) from the
[Wav2Lip repo](https://github.com/Rudrabha/Wav2Lip) and put them in `checkpoints/`.

## 1. Convert

```bash
python scripts/convert.py --checkpoint checkpoints/wav2lip_gan.pth --batch 1 --precision fp32 int8w
```

| precision | size | what it is | use it for |
|---|---|---|---|
| `fp32` | 145 MB | float reference | **GPU (Metal)**. The GPU computes in FP16 anyway, so no separate FP16 file is needed |
| `int8w` | 37 MB | INT8 weights, float activations | smaller download/memory; ~same quality, same GPU speed |
| `int8` | 37 MB | INT8 weights + activations, calibrated | **CPU / NPU**. ~4× faster than FP32 on CPU; check quality |

`--batch N` bakes a fixed batch size into the model (GPU backends don't like dynamic shapes).
Batch 1 is lowest latency; 8–16 gives the best throughput for offline video.

### Full INT8: calibrate on real faces

INT8 activations quantize the mouth pixels directly, so calibrate on your own footage, not noise:

```bash
python infer.py --face me.mp4 --audio speech.wav --model models/wav2lip_gan_b1_fp32.tflite \
    --dump-calib calib.npz
python scripts/convert.py --checkpoint checkpoints/wav2lip_gan.pth --precision int8 --calib calib.npz
python scripts/compare.py --checkpoint checkpoints/wav2lip_gan.pth --calib calib.npz \
    models/wav2lip_gan_b1_*.tflite --accelerator cpu
```

`compare.py` reports error vs PyTorch, PSNR and ms/frame. Above ~40 dB is indistinguishable on a 96×96 crop;
below ~30 dB expect banding or colour shift around the mouth.

## 2. Run

```bash
python infer.py --face input.mp4 --audio speech.wav \
    --model models/wav2lip_gan_b1_fp32.tflite --out result.mp4
```

`--face` can be a video (looped to the audio length) or a still image. BlazeFace downloads to
`models/` on first run. At the end it prints a time split (detect / Wav2Lip / everything else) so you
can see where the time goes.

Tuning:
- `--pad TOP BOTTOM LEFT RIGHT`: box padding as a fraction of the face box (default `0 0.05 0 0`).
  Keep it small: Wav2Lip expects the mouth at a fixed spot in its 96×96 crop, and with too much
  padding it draws the new lips on the chin. If lips look smeared or doubled, reduce the bottom pad.
- `--smooth 0.6`: EMA on the box between frames. Raise it if the mouth jitters, lower it for fast head motion.
- `--accelerator gpu|cpu|npu`: falls back to CPU automatically if the accelerator isn't available.

## How the conversion works

- `w2l/model.py` re-implements the Wav2Lip generator with the original module names so checkpoints load directly.
- `make_exportable()` folds all BatchNorms into the convs and rewrites the decoder's
  `ConvTranspose2d(stride=2, padding=1, output_padding=1)` layers as an unpadded transposed conv plus a crop.
  The original form fails to legalize in the LiteRT converter; the rewrite is numerically identical
  (max diff ~6e-8) and lowers to plain `TRANSPOSE_CONV`.
- `litert_torch.to_channel_last_io` makes the model's inputs/outputs NHWC, so the graph has no `TRANSPOSE` ops
  and the runtime feeds OpenCV arrays straight in.
- Converted graph: `CONV_2D`, `TRANSPOSE_CONV`, `ADD`, `CONCATENATION`, `SLICE`, `PAD`, `LOGISTIC` (all GPU-supported ops).

Model I/O:

| tensor | shape | contents |
|---|---|---|
| `args_0` | (B, 80, 16, 1) | mel window |
| `args_1` | (B, 96, 96, 6) | lower-half-masked BGR crop ‖ reference BGR crop, / 255 |
| `output_0` | (B, 96, 96, 3) | BGR, 0–1 |

## Benchmarks

Measured on a CPU-only Linux sandbox with the real `wav2lip_gan` weights (full tables in
[BENCHMARKS.md](BENCHMARKS.md)):

| model | size | Wav2Lip ms/frame | end-to-end fps | PSNR vs PyTorch |
|---|---:|---:|---:|---:|
| `fp32` | 145 MB | 90–97 | 9.3 | 139 dB |
| `int8w` | 37 MB | 94 | – | 55 dB |
| `int8` (calibrated) | 37 MB | 18–21 | 32.2 | 46 dB |

Metal (M1) numbers still to be added.

## Status

Verified on Linux (CPU only) with the real `wav2lip_gan` weights: conversion at every precision, output matching
PyTorch, and a full lip-sync run on a real clip (lips checked frame by frame). **Not yet verified:** Metal GPU speed.
Run `compare.py --accelerator gpu` on the Mac to get it.

## License

This code is MIT. The Wav2Lip **weights** are released by their authors for research/non-commercial use only;
check their license before shipping anything built on them.
