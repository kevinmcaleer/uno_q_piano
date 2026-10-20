"""Map camera pixels to positions on the printed sheet using ArUco markers.

Each camera gets a homography between the paper plane (mm) and its image
(pixels). The homography is only exact for points lying on the paper, which
is what we want for the top-down camera. For the front camera it tells us
where the paper surface is in the image, which step 3 uses to judge presses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .layout import Key, Layout


def _dictionary(name: str):
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


class MarkerDetector:
    """ArUco detection that works on OpenCV 4.6 and on 4.7+ (new API)."""

    def __init__(self, dict_name: str):
        self.dictionary = _dictionary(dict_name)
        if hasattr(cv2.aruco, "ArucoDetector"):
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self._detector = cv2.aruco.ArucoDetector(self.dictionary, params)
        else:
            self._detector = None
            self._params = cv2.aruco.DetectorParameters_create()
            self._params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    def detect(self, image: np.ndarray) -> dict[int, np.ndarray]:
        """Marker id -> 4x2 array of corner pixels (TL, TR, BR, BL)."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self._params)
        if ids is None:
            return {}
        return {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners)}


def generate_marker(dict_name: str, marker_id: int, pixels: int) -> np.ndarray:
    dictionary = _dictionary(dict_name)
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, pixels)
    return cv2.aruco.drawMarker(dictionary, marker_id, pixels)


@dataclass
class Calibration:
    role: str
    image_size: tuple[int, int]  # width, height
    mm_to_px: np.ndarray  # 3x3 homography
    markers_seen: list[int]
    reprojection_error_px: float

    @property
    def px_to_mm(self) -> np.ndarray:
        return np.linalg.inv(self.mm_to_px)

    def to_sheet(self, points_px) -> np.ndarray:
        pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.px_to_mm).reshape(-1, 2)

    def to_image(self, points_mm) -> np.ndarray:
        pts = np.asarray(points_mm, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.mm_to_px).reshape(-1, 2)

    def key_at_pixel(self, layout: Layout, x: float, y: float) -> Key | None:
        sx, sy = self.to_sheet([(x, y)])[0]
        return layout.key_at(sx, sy)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "role": self.role,
                    "image_size": list(self.image_size),
                    "mm_to_px": self.mm_to_px.tolist(),
                    "markers_seen": self.markers_seen,
                    "reprojection_error_px": self.reprojection_error_px,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        d = json.loads(Path(path).read_text())
        return cls(
            role=d["role"],
            image_size=tuple(d["image_size"]),
            mm_to_px=np.array(d["mm_to_px"], dtype=np.float64),
            markers_seen=d["markers_seen"],
            reprojection_error_px=d["reprojection_error_px"],
        )


def calibrate_from_detections(
    layout: Layout, detections: dict[int, np.ndarray], image_size: tuple[int, int], role: str
) -> Calibration | None:
    """Fit the sheet-to-image homography from whichever markers were seen.

    One marker (4 corners) is the minimum; all four give a far steadier fit
    because the corners are spread across the whole sheet.
    """
    known = sorted(i for i in detections if i in layout.markers)
    if not known:
        return None
    src = np.array([c for i in known for c in layout.markers[i]], dtype=np.float64)
    dst = np.array([c for i in known for c in detections[i]], dtype=np.float64)
    method = cv2.RANSAC if len(known) > 1 else 0
    H, _ = cv2.findHomography(src, dst, method, 3.0)
    if H is None:
        return None
    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    error = float(np.sqrt(np.mean(np.sum((projected - dst) ** 2, axis=1))))
    return Calibration(role, image_size, H, known, round(error, 3))


def draw_overlay(image: np.ndarray, layout: Layout, calib: Calibration, highlight: Key | None = None):
    """Draw the key outlines onto a camera frame so you can check the fit."""
    out = image.copy()
    for key in sorted(layout.keys, key=lambda k: k.black):
        poly = calib.to_image(key.polygon()).astype(np.int32)
        colour = (40, 40, 220) if key.black else (40, 200, 40)
        if highlight is not None and key.note == highlight.note:
            cv2.fillPoly(out, [poly], (0, 200, 255))
        cv2.polylines(out, [poly], True, colour, 1, cv2.LINE_AA)
        if not key.black and key.name.startswith("C"):
            cx, cy = calib.to_image([(key.centre()[0], key.rect[3] - 8)])[0]
            cv2.putText(out, key.name, (int(cx) - 10, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    for i in calib.markers_seen:
        poly = calib.to_image(layout.markers[i]).astype(np.int32)
        cv2.polylines(out, [poly], True, (255, 120, 0), 2, cv2.LINE_AA)
    return out
