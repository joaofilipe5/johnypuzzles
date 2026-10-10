"""
flagOr :: monochrome brasão emblem particles.

Extracts ONLY the central coat of arms (armillary sphere + shield) from the flag,
as a single-colour glyph cloud — the structure (ring/shield lines + the bright
gold/white solids), with the flat green/red field dropped. No colour is stored;
the web layer paints it in the theme's ink, so it works on light OR dark.

Output: artifacts/emblem_particles.json  +  artifacts/emblem_preview.png
"""
from __future__ import annotations
import puzzle_secrets as _secrets
import json, os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "artifacts", "flag_hi.png")
GLYPHS = ".:-=+*oxAL%#&8"


def build(cols: int = 104, seed: int = _secrets.SEED):
    random.seed(seed); np.random.seed(seed)
    img = Image.open(SRC).convert("RGB")
    W, H = img.size
    cx, cy = int(0.40 * W), int(0.50 * H)          # brasão sits on the division, vertically centred
    half = int(0.315 * H)
    box = img.crop((cx - half, cy - half, cx + half, cy + half))
    bw, bh = box.size
    rows = max(8, int(cols * (bh / bw) * 0.5))       # 0.5 ~ monospace cell aspect
    small = box.resize((cols, rows), Image.BOX)
    lum = small.convert("L")
    edge = np.asarray(lum.filter(ImageFilter.FIND_EDGES), float) / 255.0
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = 0.0   # kill FIND_EDGES border frame
    ln = np.asarray(lum, float) / 255.0
    ng = len(GLYPHS)
    hcx, hcy = (cols - 1) / 2.0, (rows - 1) / 2.0

    parts = []
    for y in range(rows):
        for x in range(cols):
            rx, ry = (x - hcx) / hcx, (y - hcy) / hcy
            if rx * rx + ry * ry > 1.0:               # clip to an ellipse -> drop corners/frame
                continue
            e = min(1.0, edge[y, x] * 1.9)
            bright = ln[y, x]                         # gold rings / white shield read bright in grey
            if e < 0.27 and bright < 0.72:            # drop the flat field + the faint division line
                continue
            inten = min(1.0, e * 0.7 + max(0.0, bright - 0.5) * 1.3)
            g = int(min(ng - 1, (0.28 + 0.72 * inten) * (ng - 1)))
            parts.append([x, y, g])

    data = {"cols": cols, "rows": rows, "glyphs": GLYPHS, "p": parts}
    os.makedirs(os.path.join(ROOT, "artifacts"), exist_ok=True)
    json.dump(data, open(os.path.join(ROOT, "artifacts", "emblem_particles.json"), "w"),
              separators=(",", ":"))
    _preview(data)
    print(f"[flagOr] emblem particles={len(parts)} grid={cols}x{rows} "
          f"-> artifacts/emblem_particles.json")
    return data


def _preview(data, cell=(10, 18)):
    cols, rows = data["cols"], data["rows"]
    cw, ch = cell
    im = Image.new("RGB", (cols * cw, rows * ch), (10, 11, 14))
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 15)
    except Exception:
        font = ImageFont.load_default()
    gl = data["glyphs"]
    for x, y, g in data["p"]:
        d.text((x * cw, y * ch - 2), gl[g], fill=(230, 232, 236), font=font)
    im.save(os.path.join(ROOT, "artifacts", "emblem_preview.png"))


if __name__ == "__main__":
    build()
