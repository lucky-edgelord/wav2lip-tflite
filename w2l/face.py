"""Face boxes from MediaPipe's Face Detector (BlazeFace short-range).

BlazeFace's box already spans roughly eyebrows to chin, which is close to the
S3FD crops Wav2Lip was trained on, so it needs only a little padding below the
chin. Too much padding moves the mouth within the 96x96 crop and the model then
draws the new lips on the chin. Boxes are smoothed over time to stop jitter.
"""

from dataclasses import dataclass

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

BLAZEFACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                 "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")


@dataclass
class BoxParams:
    # Padding as a fraction of box height/width: (top, bottom, left, right).
    pad: tuple[float, float, float, float] = (0.0, 0.05, 0.0, 0.0)
    smooth: float = 0.6          # EMA weight on the previous box (0 = no smoothing)
    min_confidence: float = 0.5


class FaceBoxer:
    def __init__(self, model_path: str, use_gpu: bool = True, params: BoxParams = BoxParams()):
        self.params = params
        self.detector = self._create(model_path, use_gpu)
        self._prev = None
        self._t = 0

    def _create(self, model_path, use_gpu):
        delegates = [mp_tasks.BaseOptions.Delegate.GPU, mp_tasks.BaseOptions.Delegate.CPU] \
            if use_gpu else [mp_tasks.BaseOptions.Delegate.CPU]
        last = None
        for d in delegates:
            try:
                opts = vision.FaceDetectorOptions(
                    base_options=mp_tasks.BaseOptions(model_asset_path=model_path, delegate=d),
                    running_mode=vision.RunningMode.VIDEO,
                    min_detection_confidence=self.params.min_confidence,
                )
                det = vision.FaceDetector.create_from_options(opts)
                self.delegate = d.name
                return det
            except Exception as e:  # GPU delegate unavailable on this build/OS
                last = e
        raise RuntimeError(f"Could not create MediaPipe face detector: {last}")

    def __call__(self, frame_bgr: np.ndarray, fps: float) -> tuple[int, int, int, int] | None:
        """Return a padded, smoothed (x1, y1, x2, y2) box for the largest face, or the last box."""
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(self._t * 1000 / fps)
        self._t += 1
        dets = self.detector.detect_for_video(image, ts).detections
        if not dets:
            return self._as_int(self._prev, w, h)

        bb = max(dets, key=lambda d: d.bounding_box.width * d.bounding_box.height).bounding_box
        pt, pb, pl, pr = self.params.pad
        box = np.array([
            bb.origin_x - pl * bb.width,
            bb.origin_y - pt * bb.height,
            bb.origin_x + bb.width * (1 + pr),
            bb.origin_y + bb.height * (1 + pb),
        ], dtype=np.float32)
        if self._prev is not None and self.params.smooth > 0:
            box = self.params.smooth * self._prev + (1 - self.params.smooth) * box
        self._prev = box
        return self._as_int(box, w, h)

    @staticmethod
    def _as_int(box, w, h):
        if box is None:
            return None
        x1, y1, x2, y2 = box
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    def close(self):
        self.detector.close()
