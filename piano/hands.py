"""Thin wrapper around the MediaPipe Tasks hand landmarker (two hands)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"
FINGERTIPS = (4, 8, 12, 16, 20)  # thumb, index, middle, ring, little


def ensure_model(path: Path = DEFAULT_MODEL) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand model to {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class HandTracker:
    """Finds up to two hands per frame. Use one tracker per camera."""

    def __init__(self, model_path: Path | None = None, num_hands: int = 2):
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp_image = __import__("mediapipe").Image
        self._image_format = __import__("mediapipe").ImageFormat.SRGB
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path or DEFAULT_MODEL))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int):
        """Returns a list of (handedness, 21x3 landmarks in pixels) per hand."""
        timestamp_ms = max(timestamp_ms, self._last_ts + 1)  # must strictly increase
        self._last_ts = timestamp_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(
            self._mp_image(image_format=self._image_format, data=rgb), timestamp_ms
        )
        h, w = frame_bgr.shape[:2]
        hands = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            pts = np.array([(p.x * w, p.y * h, p.z * w) for p in landmarks], dtype=np.float32)
            hands.append((handedness[0].category_name, pts))
        return hands

    def close(self) -> None:
        self._landmarker.close()
