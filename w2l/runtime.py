"""Thin wrapper over ai-edge-litert's CompiledModel for the Wav2Lip generator.

Handles float models and fully quantized ones (int8 / int16 I/O) the same way:
callers always pass and receive float32.

Batching: the batch size is baked into the model at conversion. On CPU, batch 8
was ~25% faster per frame than batch 1 (BENCHMARKS.md); batch 1 gives the lowest
latency for live use.
"""

import numpy as np
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.hardware_accelerator import HardwareAccelerator
from ai_edge_litert.interpreter import Interpreter

SIG = "serving_default"
MEL, FACE, OUT = "args_0", "args_1", "output_0"


class _Quant:
    """float <-> integer conversion for one tensor (identity for float tensors)."""

    def __init__(self, dtype, scale, zero_point):
        self.dtype = np.dtype(dtype)
        self.scale, self.zp = float(scale), int(zero_point)
        self.is_int = self.dtype.kind == "i" and self.scale > 0

    def to_model(self, x):
        if not self.is_int:
            return np.ascontiguousarray(x, np.float32)
        info = np.iinfo(self.dtype)
        q = np.round(x / self.scale) + self.zp
        return np.clip(q, info.min, info.max).astype(self.dtype)

    def from_model(self, q):
        return q.astype(np.float32) if not self.is_int else (q.astype(np.float32) - self.zp) * self.scale


def _io_quant(path):
    runner = Interpreter(model_path=path).get_signature_runner(SIG)
    q = {}
    for name, d in {**runner.get_input_details(), **runner.get_output_details()}.items():
        scale, zp = d["quantization"]
        q[name] = _Quant(d["dtype"], scale, zp)
    return q


class Wav2LipLiteRT:
    def __init__(self, model_path: str, accelerator: str = "gpu"):
        self.model, self.accelerator = self._load(model_path, accelerator)
        details = self.model.get_input_tensor_details(SIG)
        self.batch = details[MEL]["shape"][0]
        self.out_shape = (self.batch, 96, 96, 3)
        self.q = _io_quant(model_path)
        self.mel_buf = self.model.create_input_buffer_by_name(SIG, MEL)
        self.face_buf = self.model.create_input_buffer_by_name(SIG, FACE)
        self.out_buf = self.model.create_output_buffer_by_name(SIG, OUT)

    @staticmethod
    def _load(path, accelerator):
        order = {"gpu": ["GPU", "CPU"], "cpu": ["CPU"], "npu": ["NPU", "CPU"]}[accelerator]
        last = None
        for name in order:
            try:
                m = CompiledModel.from_file(path, hardware_accel=getattr(HardwareAccelerator, name))
                if name != "CPU" and not m.is_fully_accelerated():
                    print(f"note: {name} is running only part of the graph; the rest is on CPU")
                return m, name
            except Exception as e:
                last = e
                print(f"note: {name} accelerator unavailable; falling back")
        raise RuntimeError(f"could not load {path}: {last}")

    def __call__(self, mel: np.ndarray, face: np.ndarray) -> np.ndarray:
        """mel (n, 80, 16, 1), face (n, 96, 96, 6), n <= batch. Returns float32 (n, 96, 96, 3)."""
        n = mel.shape[0]
        if n < self.batch:  # the batch size is baked in, so pad the last partial batch
            mel = np.concatenate([mel, np.zeros((self.batch - n, *mel.shape[1:]), np.float32)])
            face = np.concatenate([face, np.zeros((self.batch - n, *face.shape[1:]), np.float32)])
        self.mel_buf.write(self.q[MEL].to_model(mel))
        self.face_buf.write(self.q[FACE].to_model(face))
        # Buffers in signature order: args_0 (mel), args_1 (face).
        self.model.run_by_index(0, [self.mel_buf, self.face_buf], [self.out_buf])
        out = self.out_buf.read(int(np.prod(self.out_shape)), self.q[OUT].dtype)
        return self.q[OUT].from_model(out).reshape(self.out_shape)[:n]
