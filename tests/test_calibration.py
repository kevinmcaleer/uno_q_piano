"""Render the sheet, view it through a fake camera, and check keys map back.

Run with:  python -m pytest tests
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from make_sheet import render_sheet  # noqa: E402
from piano.calibration import MarkerDetector, calibrate_from_detections  # noqa: E402
from piano.layout import Layout, build_layout, parse_note  # noqa: E402

DPI = 100
PX_PER_MM = DPI / 25.4


@pytest.fixture(scope="module")
def layout():
    return build_layout("a3", parse_note("C3"), 2)


@pytest.fixture(scope="module")
def sheet(layout):
    return np.array(render_sheet(layout, DPI).convert("RGB"))[:, :, ::-1].copy()


def fake_camera(sheet, dst_corners, size=(1280, 720)):
    """Warp the flat sheet image as if seen by a camera at an angle."""
    h, w = sheet.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    H = cv2.getPerspectiveTransform(src, np.float32(dst_corners))
    img = cv2.warpPerspective(sheet, H, size, borderValue=(90, 90, 90))
    return img, H


VIEWS = {
    # Roughly overhead, slightly rotated.
    "top": [[180, 60], [1110, 90], [1090, 680], [160, 650]],
    # Low front view: far edge squashed, near edge wide.
    "front": [[330, 250], [950, 250], [1240, 600], [40, 600]],
}


@pytest.mark.parametrize("view", VIEWS)
def test_every_key_maps_back(layout, sheet, view):
    img, H_sheetpx_to_img = fake_camera(sheet, VIEWS[view])
    detections = MarkerDetector(layout.aruco_dict).detect(img)
    assert sorted(detections) == [0, 1, 2, 3]

    calib = calibrate_from_detections(layout, detections, (img.shape[1], img.shape[0]), view)
    assert calib is not None
    assert calib.reprojection_error_px < 1.5

    for key in layout.keys:
        # Where the key centre really is in the image, from the true warp.
        cx, cy = key.centre()
        if not key.black:
            cy = key.rect[3] - 15  # near end of a white key, away from the black keys
        true_px = cv2.perspectiveTransform(
            np.float64([[[cx * PX_PER_MM, cy * PX_PER_MM]]]), H_sheetpx_to_img
        )[0, 0]
        found = calib.key_at_pixel(layout, *true_px)
        assert found is not None and found.note == key.note, f"{key.name} read as {found}"


def test_partial_markers_still_calibrate(layout, sheet):
    img, _ = fake_camera(sheet, VIEWS["top"])
    detections = MarkerDetector(layout.aruco_dict).detect(img)
    calib = calibrate_from_detections(layout, {0: detections[0]}, (1280, 720), "top")
    assert calib is not None and calib.markers_seen == [0]


def test_layout_round_trip(layout, tmp_path):
    path = tmp_path / "layout.json"
    layout.save(path)
    loaded = Layout.load(path)
    assert [k.note for k in loaded.keys] == [k.note for k in layout.keys]
    assert loaded.key_at(*layout.key_for_note(61).centre()).name == "C#4"
    assert parse_note("C4") == 60 and parse_note("A#3") == 58
