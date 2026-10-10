"""
flagOr :: extract colored glyph-particles from the flag raster.

Turns the crisp resvg flag PNG into a sparse cloud of monospace glyphs (the
"thinking-orb" particle look): flat green/red fields become sparse coloured
particles, the golden armillary sphere + shield become dense detailed glyphs.

Outputs:
  artifacts/flag_particles.json   (compact data the canvas animation consumes)
  artifacts/particles_preview.png (static sanity frame, Menlo on black)
"""
from __future__ import annotations
import puzzle_secrets as _secrets
import json, os, random
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

SRC = "artifacts/flag_hi.png"
GLYPHS = list(".:-=+*oxAL%#&8")            # dark -> dense
# boosted palette (readable on black): green, red, gold, white, blue
PAL = [(26,158,90), (239,51,64), (255,207,63), (240,240,240), (43,79,216)]
PAL_NP = np.array(PAL, dtype=float)

def quantize(rgb):
    d = ((PAL_NP[None,None,:,:] - rgb[:,:,None,:].astype(float))**2).sum(-1)
    return d.argmin(-1)                      # index into PAL

def build(cols=240, seed=_secrets.SEED):
    random.seed(seed); np.random.seed(seed)
    img = Image.open(SRC).convert("RGB")
    W, H = img.size
    rows = int(round(cols * (H / W) * 0.52))  # 0.52 ~ monospace cell aspect
    small = img.resize((cols, rows), Image.BOX)
    rgb = np.asarray(small)
    # edges on luminance
    lum = Image.fromarray(rgb).convert("L")
    edge = np.asarray(lum.filter(ImageFilter.FIND_EDGES).resize((cols, rows))) / 255.0
    edge = np.clip(edge * 2.2, 0, 1)
    ln = np.asarray(lum, float) / 255.0
    cidx = quantize(rgb)

    parts = []
    for y in range(rows):
        for x in range(cols):
            c = int(cidx[y, x]); e = float(edge[y, x])
            is_field = c in (0, 1) and e < 0.25            # flat green/red
            if random.random() > 0.99:                     # print the WHOLE flag (~1% gaps = texture)
                continue
            g = int(min(len(GLYPHS) - 1, int((0.35 + 0.65 * ln[y, x]) * (len(GLYPHS) - 1))))
            if is_field:
                g = random.randint(1, 4)                   # faint glyphs so fields stay legible as fields
            parts.append((x, y, c, g))
    data = {"cols": cols, "rows": rows, "glyphs": "".join(GLYPHS),
            "pal": PAL, "p": parts}
    os.makedirs("artifacts", exist_ok=True)
    json.dump(data, open("artifacts/flag_particles.json", "w"), separators=(",", ":"))
    _preview(data)
    print(f"[flagOr] particles={len(parts)} grid={cols}x{rows} -> artifacts/flag_particles.json")
    return data

def _preview(data, cell=(9, 16), scale=1):
    cols, rows = data["cols"], data["rows"]
    cw, ch = cell
    im = Image.new("RGB", (cols * cw, rows * ch), (8, 9, 12))
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
    gl = data["glyphs"]; pal = data["pal"]
    for x, y, c, g in data["p"]:
        d.text((x * cw, y * ch - 2), gl[g], fill=tuple(pal[c]), font=font)
    im.save("artifacts/particles_preview.png")

if __name__ == "__main__":
    build()
