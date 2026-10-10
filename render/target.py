"""
flagOr :: A0 -- ground-truth targets (LOCAL, runs now).

This is the small local prerequisite that unblocks the DelftBlue authoring (Phase A1).
It is NOT the pretty terminal renderer (that is Phase B / render/flag.py).

It produces, for each Bandit-style level, the *binary* image that the level's max-cut
optimum must reproduce -- because max-cut yields a 2-colouring, every target here is a
2-partition (foreground vs background):

  L2  field   : green (0)  vs red (1)           -- the 2/5 hoist split
  L3  sphere  : gold  (1)  vs ground (0)        -- the armillary-sphere silhouette
  L4  word    : ink   (1)  vs shield-field (0)  -- the hidden pixel-font word (the answer)

It also renders a full-colour PT flag PNG for eyeballing / the Phase-B reveal reference.

Usage:
    python -m render.target --H 64 --word $JP_CODE --out artifacts/targets
Outputs (under --out): *.npy (uint8 {0,1}) + *.png previews + flag_color.png + montage.png
"""
from __future__ import annotations

import puzzle_secrets as _secrets
import argparse
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---- Official-ish Portuguese flag palette (RGB) ---------------------------------
GREEN = (0, 98, 51)     # Pantone 349-ish
RED = (218, 41, 28)     # Pantone 485-ish
GOLD = (255, 204, 41)
WHITE = (255, 255, 255)
BLUE = (0, 51, 153)
GROUND = (0, 0, 0)      # "not-gold" background for the binary sphere target

ASPECT = 1.5            # width : height = 3 : 2
SPLIT = 0.40            # green field = left 2/5 of the width
SPHERE_R = 0.30         # sphere outer radius as a fraction of flag height


# ---- geometry helpers -----------------------------------------------------------
def _dims(H: int) -> tuple[int, int]:
    """Return (W, H) for the flag canvas at hoist height H."""
    return int(round(ASPECT * H)), H


def _sphere_center(H: int) -> tuple[float, float]:
    W, _ = _dims(H)
    return SPLIT * W, H / 2.0          # centred ON the green/red division, vertically centred


def make_flag_rgb(H: int = 400) -> np.ndarray:
    """Full-colour PT flag (preview + Phase-B reveal reference). Returns HxWx3 uint8."""
    W, _ = _dims(H)
    img = Image.new("RGB", (W, H), RED)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, int(SPLIT * W), H], fill=GREEN)      # green hoist field

    cx, cy = _sphere_center(H)
    R = SPHERE_R * H
    lw = max(1, int(round(H / 90)))

    # armillary sphere: outer ring + equator + two meridians + ecliptic (tilted) + tropics
    def ell(rx, ry, **kw):
        d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=GOLD, width=lw, **kw)

    ell(R, R)                                   # outer great circle
    ell(R, R * 0.32)                            # equator (thin horizontal ellipse)
    ell(R * 0.32, R)                            # meridian 1 (thin vertical ellipse)
    ell(R * 0.70, R * 0.94)                     # meridian 2
    d.line([cx - R, cy - R * 0.55, cx + R, cy + R * 0.55], fill=GOLD, width=lw)  # ecliptic band
    for f in (-0.55, 0.55):                     # tropics
        d.line([cx - R * 0.83, cy + R * f, cx + R * 0.83, cy + R * f], fill=GOLD, width=lw)

    # white shield (escudo) centred on the sphere
    sh_h = 0.42 * R * 2
    sh_w = sh_h * 0.72
    _draw_shield(d, cx, cy, sh_w, sh_h, lw)
    return np.asarray(img, dtype=np.uint8)


def _draw_shield(d: ImageDraw.ImageDraw, cx: float, cy: float, w: float, h: float, lw: int):
    """A rough Portuguese escudo: white field, red border, 5 blue quinas in a cross."""
    left, right = cx - w / 2, cx + w / 2
    top = cy - h / 2
    body_bottom = cy + h * 0.18
    tip = cy + h / 2
    # shield outline (rounded top corners + pointed base) as a polygon
    pts = [(left, top), (right, top), (right, body_bottom), (cx, tip), (left, body_bottom)]
    d.polygon(pts, fill=WHITE, outline=RED)
    for _ in range(max(1, lw)):                 # thicken border
        d.line(pts + [pts[0]], fill=RED, width=lw + 1)
    # 5 quinas (blue escutcheons) in a plus arrangement
    q = w * 0.16
    offs = [(0, -h * 0.18), (0, h * 0.02), (-w * 0.22, -h * 0.08),
            (w * 0.22, -h * 0.08), (0, h * 0.22)]
    for dx, dy in offs:
        qx, qy = cx + dx, cy + dy
        d.rectangle([qx - q / 2, qy - q / 2, qx + q / 2, qy + q * 0.75], fill=BLUE)


# ---- binary level targets -------------------------------------------------------
def level_field_target(H: int) -> np.ndarray:
    """L2: 0 = green hoist field (left 2/5), 1 = red fly field."""
    W, _ = _dims(H)
    t = np.ones((H, W), dtype=np.uint8)
    t[:, : int(round(SPLIT * W))] = 0
    return t


def level_sphere_target(H: int) -> np.ndarray:
    """L3: 1 = golden armillary-sphere pixels, 0 = ground. Derived from the colour render."""
    rgb = make_flag_rgb(H)
    r, g, b = rgb[..., 0].astype(int), rgb[..., 1].astype(int), rgb[..., 2].astype(int)
    gold = (r > 180) & (g > 140) & (b < 120)     # gold-ish
    return gold.astype(np.uint8)


def render_word_binary(word: str, height_px: int) -> np.ndarray:
    """Render WORD in a chunky pixel font -> binary array (1 = ink), height ~= height_px."""
    font = ImageFont.load_default()
    tmp = Image.new("L", (max(8, 8 * len(word)), 16), 0)
    ImageDraw.Draw(tmp).text((0, 2), word, fill=255, font=font)
    bbox = tmp.getbbox() or (0, 0, 8, 8)
    crop = tmp.crop(bbox)
    scale = max(1, int(round(height_px / crop.height)))
    big = crop.resize((crop.width * scale, crop.height * scale), Image.NEAREST)
    return (np.asarray(big) > 127).astype(np.uint8)


def level_word_target(H: int, word: str) -> np.ndarray:
    """L4: a shield panel; 1 = ink (the hidden word), 0 = white shield field."""
    W, _ = _dims(H)
    panel = np.zeros((H, W), dtype=np.uint8)
    ink = render_word_binary(word, height_px=max(7, int(H * 0.35)))
    ih, iw = ink.shape
    if iw > W - 4:                                # squeeze to fit width if needed
        s = (W - 4) / iw
        ink = np.asarray(
            Image.fromarray(ink * 255).resize((W - 4, max(1, int(ih * s))), Image.NEAREST)
        )
        ink = (ink > 127).astype(np.uint8)
        ih, iw = ink.shape
    y0, x0 = (H - ih) // 2, (W - iw) // 2
    panel[y0 : y0 + ih, x0 : x0 + iw] = ink
    return panel


# ---- io -------------------------------------------------------------------------
def _save_binary_png(arr: np.ndarray, path: str):
    Image.fromarray((arr * 255).astype(np.uint8), mode="L").save(path)


def _montage(images: list[np.ndarray], path: str, scale: int = 4):
    """Stack binary targets vertically (upscaled) into one preview PNG."""
    W = max(a.shape[1] for a in images)
    rows = []
    for a in images:
        pad = np.zeros((a.shape[0], W), dtype=np.uint8)
        pad[:, : a.shape[1]] = a
        rows.append(pad)
        rows.append(np.zeros((2, W), dtype=np.uint8))     # separator
    stack = (np.vstack(rows) * 255).astype(np.uint8)
    img = Image.fromarray(stack, "L").resize(
        (W * scale, stack.shape[0] * scale), Image.NEAREST
    )
    img.save(path)


def build(H: int, word: str, out: str) -> dict[str, np.ndarray]:
    os.makedirs(out, exist_ok=True)
    targets = {
        "L2_field": level_field_target(H),
        "L3_sphere": level_sphere_target(H),
        "L4_word": level_word_target(H, word),
    }
    for name, arr in targets.items():
        np.save(os.path.join(out, f"{name}.npy"), arr)
        _save_binary_png(arr, os.path.join(out, f"{name}.png"))
    Image.fromarray(make_flag_rgb(max(400, H * 6))).save(os.path.join(out, "flag_color.png"))
    _montage(list(targets.values()), os.path.join(out, "montage.png"))
    return targets


def main():
    ap = argparse.ArgumentParser(description="flagOr A0 target generator")
    ap.add_argument("--H", type=int, default=64, help="CO target hoist height (px)")
    ap.add_argument("--word", type=str, default=_secrets.CODE, help="hidden word for L4 (the answer)")
    ap.add_argument("--out", type=str, default="artifacts/targets")
    a = ap.parse_args()
    targets = build(a.H, a.word.upper(), a.out)
    print(f"[flagOr] targets written to {a.out}/  (H={a.H}, word={a.word.upper()!r})")
    for name, arr in targets.items():
        frac = 100.0 * arr.mean()
        print(f"  {name:10s} shape={arr.shape}  foreground={frac:5.1f}%")
    print("  previews: flag_color.png, montage.png")


if __name__ == "__main__":
    main()
