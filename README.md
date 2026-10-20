# uno_q_piano

Play a piano printed on paper. Two cameras watch your hands: a top-down camera
works out which key each fingertip is over, and a low front camera sees the
moment a finger touches the paper. Notes go out as network MIDI to a DAW
(Logic Pro on a Mac). Runs on an Arduino UNO Q or Ventuno Q.

Status: proof of concept, built in four steps.

1. **Rig and calibration** (this step): printable sheet, camera calibration, frame-rate test.
2. Notes from the top view: MediaPipe tracks all 10 fingertips and maps them to keys.
3. Presses from the front view: fingertip touching the paper means note on.
4. MIDI into Logic over Apple Network MIDI (RTP-MIDI).

## Install

On the board's Linux side (or a laptop for testing):

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests          # checks calibration on synthetic camera views
```

MediaPipe downloads its hand model (about 8 MB) into `models/` the first time
it runs. If it fails with `libEGL.so.1` missing, install it with
`sudo apt install libegl1 libgles2`.

## 1. Print the sheet

```sh
python make_sheet.py --paper a3 --start C3 --octaves 2
```

This writes `sheet.pdf`, a `sheet.png` preview and `sheet_layout.json`. The
default is two octaves (C3 to C5) on A3 with real-size 23 mm keys. On A4 the
keys shrink to about 18 mm to fit. Print at **100% / actual size**, not
"fit to page", then check a white key with a ruler against the width printed
at the bottom of the sheet. Tape the sheet flat; curling paper bends the fit.

The four ArUco markers in the corners are how the cameras find the sheet.
Keep them uncovered and don't trim them off.

## 2. Mount the cameras

* **Top camera:** straight down over the middle of the keyboard, high enough
  to see all four markers (about 50 to 70 cm for A3). Overhead light that
  doesn't cast hand shadows across the keys works best.
* **Front camera:** in front of the sheet, low, about 15 to 25 degrees above
  the table and 30 to 50 cm back, so it sees the whole width of the keys and
  the black strip along the near edge. It must see at least the two near
  markers (ids 2 and 3); all four is better.

USB webcams are the easy option. The UNO Q has one USB-C port, so two
webcams need a powered USB-C hub. List the cameras with
`v4l2-ctl --list-devices` (package `v4l-utils`); each webcam usually shows up
as two `/dev/video*` nodes and the first one is the one to use.

## 3. Calibrate each camera

With a screen attached (or a laptop with the cameras plugged in):

```sh
python calibrate.py --role top --camera 0
python calibrate.py --role front --camera 2
```

The key outlines are drawn over the live image. Hover the mouse to see which
key it reads, press `s` to save and `q` to quit.

Headless over SSH:

```sh
python calibrate.py --role top --camera /dev/video0 --headless
```

This keeps the best fit from a few seconds of frames and writes
`calib_top.json` plus `calib_top_overlay.jpg`. Copy the image off and check
the outlines sit on the printed keys. A reprojection error under about 1 px
is good. Re-run calibration whenever a camera or the sheet moves.

## 4. Choose the board

Run the same benchmark on each board with both cameras plugged in:

```sh
python bench_cameras.py --cams /dev/video0 /dev/video2 --process hands --seconds 20
```

It prints frames per second for each camera while MediaPipe tracks two hands
on both, which is the real workload for steps 2 and 3. Try `--process none`
too, to see whether USB capture alone is the limit. Aim for about 25 fps or
more on both cameras; if neither board gets there, try
`--width 424 --height 240` or `--hands-width 480`.

## Files

| File | What it does |
| --- | --- |
| `make_sheet.py` | Printable sheet and `sheet_layout.json` |
| `calibrate.py` | Fits each camera to the sheet, saves `calib_<role>.json` |
| `bench_cameras.py` | Two-camera frame-rate test with optional hand tracking |
| `piano/layout.py` | Key positions on the sheet and point-to-key lookup |
| `piano/calibration.py` | ArUco detection and the pixel-to-sheet mapping |
| `piano/hands.py` | MediaPipe two-hand tracker wrapper |
