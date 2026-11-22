"""Check converted models against PyTorch: error, PSNR and latency per file.

    python scripts/compare.py --checkpoint checkpoints/wav2lip_gan.pth \
        models/wav2lip_gan_b1_*.tflite --accelerator gpu [--calib calib.npz]

With --calib, inputs are real face/mel samples (much more telling than noise).
PSNR above ~40 dB is visually indistinguishable on a 96x96 mouth crop;
below ~30 dB expect visible banding or colour shift.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from w2l.model import Wav2Lip, load_wav2lip  # noqa: E402
from w2l.runtime import Wav2LipLiteRT  # noqa: E402


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--checkpoint", help="omit to compare against seed-0 random weights")
    ap.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu", "npu"])
    ap.add_argument("--calib", help="npz of real inputs from infer.py --dump-calib")
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()

    if args.checkpoint:
        ref_model = load_wav2lip(args.checkpoint)
    else:
        torch.manual_seed(0)
        ref_model = Wav2Lip().eval()

    for path in args.models:
        rt = Wav2LipLiteRT(path, args.accelerator)
        b = rt.batch
        if args.calib:
            d = np.load(args.calib)
            mel, face = d["mel"][:b].astype(np.float32), d["face"][:b].astype(np.float32)
        else:
            rng = np.random.default_rng(1)
            mel = rng.uniform(-4, 4, (b, 80, 16, 1)).astype(np.float32)
            face = rng.uniform(0, 1, (b, 96, 96, 6)).astype(np.float32)

        with torch.no_grad():
            ref = ref_model(torch.from_numpy(mel).permute(0, 3, 1, 2),
                            torch.from_numpy(face).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).numpy()
        out = rt(mel, face)
        for _ in range(3):
            rt(mel, face)
        t = time.perf_counter()
        for _ in range(args.runs):
            rt(mel, face)
        ms = (time.perf_counter() - t) / args.runs * 1000

        print(f"{os.path.basename(path):40s} {rt.accelerator:3s} "
              f"{os.path.getsize(path) / 1e6:6.1f} MB  max|err| {np.abs(out - ref).max():.4f}  "
              f"PSNR {psnr(out, ref):5.1f} dB  {ms:7.1f} ms/batch  {ms / b:6.1f} ms/frame")


if __name__ == "__main__":
    main()
