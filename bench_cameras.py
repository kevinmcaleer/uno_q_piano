"""Measure how fast a board can run both cameras, to pick UNO Q or Ventuno Q.

    python bench_cameras.py --cams 0 2 --process hands --seconds 20

Each camera runs in its own thread: grab a frame, then optionally process it.
  --process none    capture only (USB bandwidth and decoding)
  --process aruco   plus marker detection (cheap)
  --process hands   plus MediaPipe two-hand tracking (what steps 2 and 3 need)

Run it once per board with the same settings and compare the numbers. For a
playable feel we want roughly 25 fps or better on both cameras with hands on.
"""

from __future__ import annotations

import argparse
import os
import platform
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from calibrate import open_camera


def cpu_temp_c() -> float | None:
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            return int(zone.read_text()) / 1000
        except (OSError, ValueError):
            continue
    return None


class CameraWorker(threading.Thread):
    def __init__(self, source: str, args):
        super().__init__(daemon=True)
        self.source, self.args = source, args
        self.frames = 0
        self.process_ms: list[float] = []
        self.hands_seen = 0
        self.error: str | None = None
        self.actual_size = None
        self.stop = threading.Event()

    def run(self) -> None:
        try:
            self._run()
        except BaseException as e:  # report it in the table instead of dying silently
            self.error = f"{type(e).__name__}: {e}"

    def _run(self) -> None:
        cap = open_camera(self.source, self.args.width, self.args.height)
        self.actual_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.set(cv2.CAP_PROP_FPS, self.args.fps)
        process, close = self._make_processor()
        t0 = time.monotonic()
        while not self.stop.is_set():
            ok, frame = cap.read()
            if not ok:
                continue
            self.frames += 1
            if process:
                start = time.perf_counter()
                process(frame, int((time.monotonic() - t0) * 1000))
                self.process_ms.append((time.perf_counter() - start) * 1000)
        cap.release()
        close()

    def _make_processor(self):
        mode = self.args.process
        if mode == "aruco":
            from piano.calibration import MarkerDetector

            detector = MarkerDetector("DICT_4X4_50")
            return (lambda frame, ts: detector.detect(frame)), lambda: None
        if mode == "hands":
            from piano.hands import HandTracker

            tracker = HandTracker()

            def run(frame, ts):
                small = frame
                if self.args.hands_width and frame.shape[1] > self.args.hands_width:
                    scale = self.args.hands_width / frame.shape[1]
                    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                self.hands_seen += len(tracker.detect(small, ts))

            return run, tracker.close
        return None, lambda: None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cams", nargs="+", default=["0", "2"], help="indices or /dev/videoN paths")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--process", choices=["none", "aruco", "hands"], default="hands")
    ap.add_argument("--hands-width", type=int, default=640, help="downscale frames to this width for hands")
    args = ap.parse_args()

    print(f"{platform.node()}  {platform.machine()}  {os.cpu_count()} cores  OpenCV {cv2.__version__}")
    print(f"{len(args.cams)} camera(s) at {args.width}x{args.height}, process={args.process}, "
          f"{args.seconds:.0f}s\n")

    workers = [CameraWorker(c, args) for c in args.cams]
    for w in workers:
        w.start()
    time.sleep(2)  # let cameras and models warm up before counting
    for w in workers:
        w.frames, w.process_ms, w.hands_seen = 0, [], 0
    start = time.monotonic()
    time.sleep(args.seconds)
    elapsed = time.monotonic() - start
    load = os.getloadavg()[0]
    for w in workers:
        w.stop.set()
    for w in workers:
        w.join(timeout=5)

    print(f"{'camera':<24}{'size':<11}{'fps':>6}{'proc ms':>10}{'p95 ms':>9}{'hands/frame':>13}")
    for w in workers:
        if w.error:
            print(f"{w.source:<24}ERROR {w.error}")
            continue
        fps = w.frames / elapsed
        ms = np.array(w.process_ms) if w.process_ms else np.array([0.0])
        size = f"{w.actual_size[0]}x{w.actual_size[1]}" if w.actual_size else "?"
        hpf = w.hands_seen / w.frames if w.frames else 0
        print(f"{w.source:<24}{size:<11}{fps:>6.1f}{ms.mean():>10.1f}{np.percentile(ms, 95):>9.1f}{hpf:>13.2f}")
    temp = cpu_temp_c()
    print(f"\nload average {load:.2f}" + (f", CPU temperature {temp:.0f} C" if temp else ""))


if __name__ == "__main__":
    main()
