"""Lip-sync a video (or still image) to an audio track with LiteRT + MediaPipe.

    python infer.py --face input.mp4 --audio speech.wav \
        --model models/wav2lip_gan_b1_fp32.tflite --out result.mp4

Pipeline per frame: MediaPipe BlazeFace box -> pad + smooth -> 96x96 crop
-> Wav2Lip (LiteRT, GPU if available) -> resize -> paste back.
"""

import argparse
import os
import subprocess
import tempfile
import time
import urllib.request

import cv2
import numpy as np

from w2l.audio import load_wav, melspectrogram, mel_chunks
from w2l.face import BLAZEFACE_URL, BoxParams, FaceBoxer
from w2l.runtime import Wav2LipLiteRT

IMG = 96


class FrameSource:
    """Yields frames forever, looping a video (as Wav2Lip does) or repeating an image."""

    def __init__(self, path, fps_for_image):
        self.path = path
        self.still = os.path.splitext(path)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        if self.still:
            self.image = cv2.imread(path)
            if self.image is None:
                raise SystemExit(f"cannot read image {path}")
            self.fps = fps_for_image
        else:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise SystemExit(f"cannot open video {path}")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0

    def read(self):
        if self.still:
            return self.image.copy()
        ok, frame = self.cap.read()
        if not ok:  # loop
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                raise SystemExit("video has no frames")
        return frame


def ensure_blazeface(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"downloading BlazeFace to {path}")
        urllib.request.urlretrieve(BLAZEFACE_URL, path)
    return path


def prepare(frame, box):
    x1, y1, x2, y2 = box
    crop = cv2.resize(frame[y1:y2, x1:x2], (IMG, IMG), interpolation=cv2.INTER_AREA)
    masked = crop.copy()
    masked[IMG // 2:] = 0  # hide the lower half; the model redraws it from audio
    return np.concatenate([masked, crop], axis=2).astype(np.float32) / 255.0


def paste(frame, box, pred):
    x1, y1, x2, y2 = box
    patch = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
    frame[y1:y2, x1:x2] = cv2.resize(patch, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--face", required=True, help="video or image of the speaker")
    ap.add_argument("--audio", required=True, help="wav, or any file ffmpeg can read")
    ap.add_argument("--model", required=True, help="converted Wav2Lip .tflite")
    ap.add_argument("--out", default="result.mp4")
    ap.add_argument("--accelerator", default="gpu", choices=["gpu", "cpu", "npu"])
    ap.add_argument("--face-model", default="models/blaze_face_short_range.tflite")
    ap.add_argument("--fps", type=float, default=25.0, help="output fps when --face is an image")
    ap.add_argument("--pad", type=float, nargs=4, default=list(BoxParams.pad),
                    metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"), help="box padding, fraction of box size")
    ap.add_argument("--smooth", type=float, default=BoxParams.smooth, help="box EMA (0=off, 0.9=heavy)")
    ap.add_argument("--dump-calib", metavar="NPZ", help="save real model inputs for INT8 calibration")
    ap.add_argument("--calib-samples", type=int, default=256)
    args = ap.parse_args()

    src = FrameSource(args.face, args.fps)
    wav = load_wav(args.audio)
    chunks = mel_chunks(melspectrogram(wav), src.fps)
    n_frames = len(chunks)
    print(f"{n_frames} frames at {src.fps:.2f} fps")

    model = Wav2LipLiteRT(args.model, args.accelerator)
    boxer = FaceBoxer(ensure_blazeface(args.face_model), use_gpu=args.accelerator == "gpu",
                      params=BoxParams(pad=tuple(args.pad), smooth=args.smooth))
    print(f"Wav2Lip on {model.accelerator} (batch {model.batch}), face detector on {boxer.delegate}")

    calib_every = max(1, n_frames // args.calib_samples)
    calib_mel, calib_face = [], []

    tmp = tempfile.TemporaryDirectory()
    silent = os.path.join(tmp.name, "video.avi")
    writer = None
    t_detect = t_model = 0.0
    t_start = time.perf_counter()

    i = 0
    while i < n_frames:
        frames, boxes, mels, faces = [], [], [], []
        while len(frames) < model.batch and i < n_frames:
            frame = src.read()
            t = time.perf_counter()
            box = boxer(frame, src.fps)
            t_detect += time.perf_counter() - t
            frames.append(frame)
            boxes.append(box)
            if box is not None:
                mels.append(chunks[i].reshape(80, 16, 1))
                faces.append(prepare(frame, box))
                if args.dump_calib and i % calib_every == 0:
                    calib_mel.append(mels[-1])
                    calib_face.append(faces[-1])
            i += 1

        if faces:
            t = time.perf_counter()
            preds = model(np.stack(mels), np.stack(faces))
            t_model += time.perf_counter() - t
            it = iter(preds)
            for frame, box in zip(frames, boxes):
                if box is not None:
                    paste(frame, box, next(it))

        for frame in frames:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"MJPG"), src.fps, (w, h))
            writer.write(frame)
        print(f"\r{i}/{n_frames}", end="", flush=True)

    writer.release()
    boxer.close()
    total = time.perf_counter() - t_start
    print(f"\ndone in {total:.1f}s ({n_frames / total:.1f} fps). "
          f"detect {t_detect / n_frames * 1000:.1f} ms/frame, "
          f"wav2lip {t_model / n_frames * 1000:.1f} ms/frame, "
          f"everything else {(total - t_detect - t_model) / n_frames * 1000:.1f} ms/frame")

    if args.dump_calib:
        np.savez_compressed(args.dump_calib, mel=np.stack(calib_mel), face=np.stack(calib_face))
        print(f"saved {len(calib_mel)} calibration samples to {args.dump_calib}")

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", args.audio,
                    "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", "-preset", "veryfast", "-c:a", "aac", "-shortest", args.out],
                   check=True)
    tmp.cleanup()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
