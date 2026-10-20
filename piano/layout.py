"""Keyboard sheet layout: where every key and ArUco marker sits on the paper.

All coordinates are in millimetres on the printed sheet. The origin is the
top-left corner of the paper, x runs to the right and y runs down towards the
player. Black keys sit at the top (far) end of the white keys, like a real
keyboard seen from above.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

PAPER_SIZES_MM = {
    "a4": (297.0, 210.0),
    "a3": (420.0, 297.0),
    "letter": (279.4, 215.9),
    "tabloid": (431.8, 279.4),
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
WHITE_PITCH_CLASSES = {0, 2, 4, 5, 7, 9, 11}
# White notes that have a black key immediately above them (C#, D#, F#, G#, A#).
HAS_SHARP = {0, 2, 5, 7, 9}

ARUCO_DICT_NAME = "DICT_4X4_50"


def note_name(midi: int) -> str:
    """MIDI note number to a name like 'C4' (middle C is 60)."""
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def parse_note(name: str) -> int:
    """'C3' or '48' to a MIDI note number."""
    if name.lstrip("-").isdigit():
        return int(name)
    for length in (2, 1):
        if name[:length].upper() in [n.upper() for n in NOTE_NAMES]:
            pc = [n.upper() for n in NOTE_NAMES].index(name[:length].upper())
            return (int(name[length:]) + 1) * 12 + pc
    raise ValueError(f"Not a note name: {name!r}")


@dataclass
class Key:
    note: int
    name: str
    black: bool
    # Axis-aligned rectangle on the sheet: x0, y0, x1, y1 in mm.
    rect: tuple[float, float, float, float]

    def contains(self, x: float, y: float) -> bool:
        x0, y0, x1, y1 = self.rect
        return x0 <= x < x1 and y0 <= y < y1

    def polygon(self) -> list[tuple[float, float]]:
        x0, y0, x1, y1 = self.rect
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def centre(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.rect
        return ((x0 + x1) / 2, (y0 + y1) / 2)


@dataclass
class Layout:
    paper_mm: tuple[float, float]
    keys: list[Key]
    # Marker id -> its four corners in ArUco order (TL, TR, BR, BL), in mm.
    markers: dict[int, list[tuple[float, float]]]
    keyboard_rect: tuple[float, float, float, float]
    aruco_dict: str = ARUCO_DICT_NAME
    meta: dict = field(default_factory=dict)

    def key_at(self, x: float, y: float) -> Key | None:
        """The key under a point on the sheet, or None if it is off the keys.

        Black keys are checked first because they sit on top of the whites.
        """
        for key in self.keys:
            if key.black and key.contains(x, y):
                return key
        for key in self.keys:
            if not key.black and key.contains(x, y):
                return key
        return None

    def key_for_note(self, note: int) -> Key | None:
        return next((k for k in self.keys if k.note == note), None)

    def to_json(self) -> dict:
        return {
            "paper_mm": list(self.paper_mm),
            "keyboard_rect": list(self.keyboard_rect),
            "aruco_dict": self.aruco_dict,
            "meta": self.meta,
            "markers": {str(i): [list(c) for c in cs] for i, cs in self.markers.items()},
            "keys": [asdict(k) for k in self.keys],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Layout":
        data = json.loads(Path(path).read_text())
        return cls(
            paper_mm=tuple(data["paper_mm"]),
            keyboard_rect=tuple(data["keyboard_rect"]),
            aruco_dict=data.get("aruco_dict", ARUCO_DICT_NAME),
            meta=data.get("meta", {}),
            markers={int(i): [tuple(c) for c in cs] for i, cs in data["markers"].items()},
            keys=[
                Key(note=k["note"], name=k["name"], black=k["black"], rect=tuple(k["rect"]))
                for k in data["keys"]
            ],
        )


def build_layout(
    paper: str = "a3",
    start_note: int = 48,
    octaves: int = 2,
    white_width: float = 23.0,
    white_length: float = 140.0,
    black_width: float = 13.0,
    black_length: float = 90.0,
    marker_size: float = 30.0,
    margin: float = 10.0,
) -> Layout:
    """Lay out `octaves` octaves plus a closing note, starting on a white note.

    Real piano keys are about 23.5 mm wide. If the keyboard does not fit the
    paper, the keys are scaled down to fit and the scale is recorded in meta.
    """
    if start_note % 12 not in WHITE_PITCH_CLASSES:
        raise ValueError("The keyboard must start on a white note")
    paper_w, paper_h = PAPER_SIZES_MM[paper.lower()]

    whites = []
    note = start_note
    while len(whites) < octaves * 7 + 1:
        if note % 12 in WHITE_PITCH_CLASSES:
            whites.append(note)
        note += 1

    # Markers sit in the four corners; the keyboard sits between the top and
    # bottom marker rows so nothing overlaps.
    usable_w = paper_w - 2 * margin
    scale = min(1.0, usable_w / (len(whites) * white_width))
    ww, bw = white_width * scale, black_width * scale
    usable_h = paper_h - 2 * (margin + marker_size + margin)
    length_scale = min(1.0, usable_h / white_length)
    wl, bl = white_length * length_scale, black_length * length_scale

    kb_w = len(whites) * ww
    x0 = (paper_w - kb_w) / 2
    y0 = (paper_h - wl) / 2

    keys: list[Key] = []
    for i, n in enumerate(whites):
        keys.append(Key(n, note_name(n), False, (x0 + i * ww, y0, x0 + (i + 1) * ww, y0 + wl)))
    for i, n in enumerate(whites[:-1]):
        if n % 12 in HAS_SHARP:
            cx = x0 + (i + 1) * ww
            keys.append(Key(n + 1, note_name(n + 1), True, (cx - bw / 2, y0, cx + bw / 2, y0 + bl)))
    keys.sort(key=lambda k: k.note)

    m, s = margin, marker_size
    corners_tl = {
        0: (m, m),  # top-left
        1: (paper_w - m - s, m),  # top-right
        2: (paper_w - m - s, paper_h - m - s),  # bottom-right
        3: (m, paper_h - m - s),  # bottom-left
    }
    markers = {
        i: [(x, y), (x + s, y), (x + s, y + s), (x, y + s)] for i, (x, y) in corners_tl.items()
    }

    return Layout(
        paper_mm=(paper_w, paper_h),
        keys=keys,
        markers=markers,
        keyboard_rect=(x0, y0, x0 + kb_w, y0 + wl),
        meta={
            "paper": paper.lower(),
            "start_note": start_note,
            "octaves": octaves,
            "key_scale": round(scale, 4),
            "white_width_mm": round(ww, 2),
        },
    )
