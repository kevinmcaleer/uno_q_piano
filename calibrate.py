"""Calibrate a camera against the printed sheet.

Live, with a window (press s to save, q to quit; hover to see the key under
the mouse):

    python calibrate.py --role top --camera 0

Headless, e.g. over SSH on the board (grabs frames for a few seconds, keeps
the best fit and saves an overlay image you can copy off and check):

    python calibrate.py --role front --camera /dev/video2 --headless

From a still photo:

    python calibrate.py --role top --image photo.jpg

Writes calib_<role>.json and calib_<role>_overlay.jpg.
"""

from __future__ import annotations

import argparse
import time

import cv2

from piano.calibration import Calibration, MarkerDetector, calibrate_from_detections, draw_overlay
from piano.layout import Layout


def open_camera(source: str, width: int, height: int) -> cv2.VideoCapture:
    src = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2) if isinstance(src, int) else cv2.VideoCapture(src)
    if not cap.isOpened():
        cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {source}")
    return cap


def fit(frame, layout: Layout, detector: MarkerDetector, role: str) -> Calibration | None:
    h, w = frame.shape[:2]
    return calibrate_from_detections(layout, detector.detect(frame), (w, h), role)


def better(a: Calibration | None, b: Calibration | None) -> bool:
    """True if a is a better fit than b: more markers first, then lower error."""
    if a is None:
        return False
    if b is None:
        return True
    return (len(a.markers_seen), -a.reprojection_error_px) > (len(b.markers_seen), -b.reprojection_error_px)


def save(calib: Calibration, frame, layout: Layout, out: str) -> None:
    calib.save(f"{out}.json")
    cv2.imwrite(f"{out}_overlay.jpg", draw_overlay(frame, layout, calib))
    print(f"Saved {out}.json and {out}_overlay.jpg  markers {calib.markers_seen}  "
          f"error {calib.reprojection_error_px:.2f} px")
    if len(calib.markers_seen) < 4:
        print("Warning: not all 4 markers were seen, so the fit may drift towards the far corners.")


def run_live(cap, layout, detector, role, out) -> None:
    mouse = [None]
    cv2.namedWindow(role)
    cv2.setMouseCallback(role, lambda ev, x, y, *_: mouse.__setitem__(0, (x, y)))
    calib = None
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        current = fit(frame, layout, detector, role)
        calib = current or calib
        view = frame
        if calib is not None:
            hover = calib.key_at_pixel(layout, *mouse[0]) if mouse[0] else None
            view = draw_overlay(frame, layout, calib, hover)
            status = f"markers {calib.markers_seen} err {calib.reprojection_error_px:.2f}px"
            if hover:
                status += f"  key {hover.name}"
        else:
            status = "no markers found"
        cv2.putText(view, status + "   s=save q=quit", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(role, view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s") and current is not None:
            save(current, frame, layout, out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--role", required=True, choices=["top", "front"])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--camera", help="index (0) or device path (/dev/video2)")
    src.add_argument("--image", help="calibrate from a still image instead")
    ap.add_argument("--layout", default="sheet_layout.json")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--headless", action="store_true", help="no window, keep the best fit")
    ap.add_argument("--seconds", type=float, default=3.0, help="headless capture time")
    ap.add_argument("--out", help="output prefix (default calib_<role>)")
    args = ap.parse_args()

    layout = Layout.load(args.layout)
    detector = MarkerDetector(layout.aruco_dict)
    out = args.out or f"calib_{args.role}"

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Could not read {args.image}")
        calib = fit(frame, layout, detector, args.role)
        if calib is None:
            raise SystemExit("No markers found in the image")
        save(calib, frame, layout, out)
        return

    cap = open_camera(args.camera, args.width, args.height)
    try:
        if not args.headless:
            run_live(cap, layout, detector, args.role, out)
            return
        best, best_frame = None, None
        end = time.time() + args.seconds
        while time.time() < end:
            ok, frame = cap.read()
            if not ok:
                continue
            calib = fit(frame, layout, detector, args.role)
            if better(calib, best):
                best, best_frame = calib, frame
        if best is None:
            raise SystemExit("No markers found. Check the sheet is in view and well lit.")
        save(best, best_frame, layout, out)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
