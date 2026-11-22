"""Convert a Wav2Lip checkpoint to LiteRT (.tflite).

    python scripts/convert.py --checkpoint checkpoints/wav2lip_gan.pth --batch 1 \
        --precision fp32 int8w

Precisions:
  fp32     Float model, the reference. On GPU, LiteRT computes in FP16 by
           default anyway, so there is no separate FP16 file.
  int8w    INT8 weights, float activations. ~1/4 file size; on GPU the weights
           are expanded back to float, so it saves memory/download, not compute.
  int8     Full INT8 weights + activations (calibrated). Fastest on CPU/NPU,
           most risk of banding around the mouth. Calibrate on real data.

Model I/O is channels-last so the runtime needs no transposes:
  args_0  mel  (B, 80, 16, 1)  float32
  args_1  face (B, 96, 96, 6)  float32, [masked BGR | reference BGR] / 255
  output_0     (B, 96, 96, 3)  float32, BGR in [0, 1]
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from w2l.model import Wav2Lip, load_wav2lip, make_exportable  # noqa: E402

MEL_IN, FACE_IN = "args_0", "args_1"


def export_float(model, batch, path):
    import litert_torch
    model = make_exportable(model)
    nhwc = litert_torch.to_channel_last_io(model, args=[0, 1], outputs=[0]).eval()
    sample = (torch.randn(batch, 80, 16, 1), torch.rand(batch, 96, 96, 6))
    litert_torch.convert(nhwc, sample).export(path)
    print(f"wrote {path}")


def calibration_set(calib_npz, batch, steps):
    """Real samples (from infer.py --dump-calib) give far better INT8 than noise."""
    if calib_npz:
        data = np.load(calib_npz)
        mels, faces = data["mel"].astype(np.float32), data["face"].astype(np.float32)
    else:
        print("warning: no --calib data; calibrating on random inputs, expect worse quality")
        rng = np.random.default_rng(0)
        n = steps * batch
        mels = rng.uniform(-4, 4, (n, 80, 16, 1)).astype(np.float32)
        faces = rng.uniform(0, 1, (n, 96, 96, 6)).astype(np.float32)
    usable = min(len(mels), steps * batch) // batch * batch
    if usable == 0:
        raise SystemExit(f"need at least {batch} calibration samples")
    return [{MEL_IN: mels[i:i + batch], FACE_IN: faces[i:i + batch]}
            for i in range(0, usable, batch)]


def quantize(float_path, out_path, precision, calib):
    from ai_edge_quantizer import quantizer, recipe
    qt = quantizer.Quantizer(float_path)
    if precision == "int8w":
        qt.load_quantization_recipe(recipe.weight_only_wi8_afp32())
        result = qt.quantize()
    else:
        qt.load_quantization_recipe(recipe.static_wi8_ai8())
        result = qt.quantize(qt.calibrate({"serving_default": calib}))
    result.export_model(out_path, overwrite=True)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", help="wav2lip.pth or wav2lip_gan.pth (omit to test with random weights)")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--batch", type=int, default=1, help="fixed batch size baked into the model")
    ap.add_argument("--precision", nargs="+", default=["fp32", "int8w"],
                    choices=["fp32", "int8w", "int8"])
    ap.add_argument("--calib", help="npz from infer.py --dump-calib (for --precision int8)")
    ap.add_argument("--calib-steps", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.checkpoint:
        model = load_wav2lip(args.checkpoint)
        name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    else:
        torch.manual_seed(0)
        model = Wav2Lip().eval()
        name = "wav2lip_random"

    base = os.path.join(args.out_dir, f"{name}_b{args.batch}")
    float_path = f"{base}_fp32.tflite"
    keep_float = "fp32" in args.precision or os.path.exists(float_path)
    export_float(model, args.batch, float_path)

    calib = None
    for p in args.precision:
        if p == "fp32":
            continue
        if p == "int8" and calib is None:
            calib = calibration_set(args.calib, args.batch, args.calib_steps)
        quantize(float_path, f"{base}_{p}.tflite", p, calib)

    if not keep_float:  # only an intermediate for this run
        os.remove(float_path)


if __name__ == "__main__":
    main()
