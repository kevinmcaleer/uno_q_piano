"""Generate the printable keyboard sheet and its layout file.

    python make_sheet.py --paper a3 --start C3 --octaves 2

Writes sheet.pdf (print at 100% / "actual size", not "fit to page"),
sheet.png (a preview) and sheet_layout.json (used by every other script).
"""

from __future__ import annotations

import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from piano.calibration import generate_marker
from piano.layout import Layout, build_layout, parse_note


def render_sheet(layout: Layout, dpi: int = 300) -> Image.Image:
    px_per_mm = dpi / 25.4

    def px(v: float) -> int:
        return int(round(v * px_per_mm))

    w_mm, h_mm = layout.paper_mm
    img = Image.new("L", (px(w_mm), px(h_mm)), 255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", px(4))
    except OSError:
        font = ImageFont.load_default()

    line = max(1, px(0.5))
    for key in layout.keys:
        x0, y0, x1, y1 = (px(v) for v in key.rect)
        if key.black:
            draw.rectangle([x0, y0, x1, y1], fill=0)
        else:
            draw.rectangle([x0, y0, x1, y1], outline=0, width=line)
    # Labels go on the near end of the white keys, clear of the black keys.
    for key in layout.keys:
        if not key.black:
            cx = px(key.centre()[0])
            ty = px(key.rect[3] - 10)
            draw.text((cx, ty), key.name, fill=0, font=font, anchor="mm")

    for marker_id, corners in layout.markers.items():
        (x0, y0), _, (x1, y1), _ = corners
        size = px(x1 - x0)
        marker = Image.fromarray(generate_marker(layout.aruco_dict, marker_id, size))
        img.paste(marker, (px(x0), px(y0)))
        draw.text((px(x0), px(y1 + 2)), f"id {marker_id}", fill=0, font=font, anchor="lt")

    # A thin strip along the near edge helps the front camera find the paper edge.
    kx0, _, kx1, ky1 = layout.keyboard_rect
    draw.rectangle([px(kx0), px(ky1 + 3), px(kx1), px(ky1 + 5)], fill=0)

    meta = layout.meta
    draw.text(
        (px(w_mm / 2), px(h_mm - 6)),
        f"uno_q_piano  {meta.get('paper', '').upper()}  print at 100%  "
        f"white key {meta.get('white_width_mm')} mm",
        fill=0,
        font=font,
        anchor="mm",
    )
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", default="a3", choices=["a4", "a3", "letter", "tabloid"])
    ap.add_argument("--start", default="C3", help="lowest note, must be white (default C3)")
    ap.add_argument("--octaves", type=int, default=2)
    ap.add_argument("--white-width", type=float, default=23.0, help="mm, shrunk to fit if needed")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default="sheet", help="output name prefix")
    args = ap.parse_args()

    layout = build_layout(args.paper, parse_note(args.start), args.octaves, args.white_width)
    layout.save(f"{args.out}_layout.json")
    img = render_sheet(layout, args.dpi)
    img.save(f"{args.out}.pdf", resolution=args.dpi)
    img.resize((img.width // 4, img.height // 4)).save(f"{args.out}.png")

    whites = sum(not k.black for k in layout.keys)
    print(f"{whites} white keys, {len(layout.keys)} keys total, "
          f"{layout.keys[0].name} to {layout.keys[-1].name}")
    print(f"White key width {layout.meta['white_width_mm']} mm "
          f"(scale {layout.meta['key_scale']:.0%} of requested)")
    print(f"Wrote {args.out}.pdf, {args.out}.png, {args.out}_layout.json")


if __name__ == "__main__":
    main()
