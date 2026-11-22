"""Build the two models the web demo loads from web/assets/, plus models.json.

    python scripts/export_web_models.py --checkpoint checkpoints/wav2lip_gan.pth --calib calib.npz

  gpu: INT8 weights, float I/O, batch 1 (WebGPU expands weights to float; 49 dB vs PyTorch on an
       unseen face). Batch 1 keeps each GPU submission small: with batch 8 an Adreno 6xx phone
       hit VK_ERROR_DEVICE_LOST (driver watchdog) on the second batch.
  cpu: full INT8, int8 I/O, batch 8 (~5x faster on CPU; 35 dB on an unseen face, faint grain)

The full-INT8 model's input/output scales are written to models.json because
LiteRT.js does not expose quantization parameters.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

from ai_edge_litert.interpreter import Interpreter

HERE = os.path.dirname(os.path.abspath(__file__))


def io_params(path):
    r = Interpreter(model_path=path).get_signature_runner()
    out = {}
    for name, d in {**r.get_input_details(), **r.get_output_details()}.items():
        scale, zp = d["quantization"]
        out[name] = {"dtype": d["dtype"].__name__, "shape": d["shape"].tolist(),
                     "scale": float(scale), "zero_point": int(zp)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--calib", required=True, help="npz from infer.py --dump-calib")
    ap.add_argument("--gpu-batch", type=int, default=1)
    ap.add_argument("--cpu-batch", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "web", "assets"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    manifest = {}
    for key, prec, batch in (("gpu", "int8w", args.gpu_batch), ("cpu", "int8", args.cpu_batch)):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [sys.executable, os.path.join(HERE, "convert.py"), "--checkpoint", args.checkpoint,
                   "--batch", str(batch), "--precision", prec, "--out-dir", tmp]
            if prec == "int8":
                cmd += ["--calib", args.calib, "--augment-calib", "--calib-steps", str(max(1, 640 // batch))]
            subprocess.run(cmd, check=True)
            src = os.path.join(tmp, f"{name}_b{batch}_{prec}.tflite")
            dst = f"{name}_b{batch}_{prec}.tflite"
            shutil.copy(src, os.path.join(args.out, dst))
            manifest[key] = {"file": dst, "precision": prec, "batch": batch,
                             "bytes": os.path.getsize(src), "io": io_params(src)}
    with open(os.path.join(args.out, "models.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps({k: (v["file"], v["bytes"]) for k, v in manifest.items()}))


if __name__ == "__main__":
    main()
