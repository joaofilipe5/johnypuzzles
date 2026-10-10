#!/usr/bin/env python3
"""
flagOr :: ANSI terminal twin of the web animation.

Renders the Portuguese flag as a field of 24-bit-colour terminal glyphs that
constantly change character (churn) and pulse in brightness (pump) — the same
motion as the site, but drawn with ANSI escapes so it runs in any truecolor
terminal. This is the seed of the real `ssh flagor.pt` Textual TUI.

Run it:
    python3 render/ansi_flag.py              # fills your terminal; Ctrl-C to quit
    python3 render/ansi_flag.py --cols 120   # force a width
    python3 render/ansi_flag.py --frames 3   # render N frames and exit (for testing)

Needs numpy + Pillow (already on this machine) and artifacts/flag_hi.png.
"""
from __future__ import annotations
import argparse, math, os, random, shutil, sys, time
import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "artifacts", "flag_hi.png")
GLYPHS = ".:-=+*oxAL%#&8"
PAL = [(26, 158, 90), (239, 51, 64), (255, 207, 63), (240, 240, 240), (43, 79, 216)]
PAL_NP = np.array(PAL, float)
DIM = 0.40  # match the site (a lot darker)

HIDE, SHOW, HOME, CLEAR, RESET = "\033[?25l", "\033[?25h", "\033[H", "\033[2J", "\033[0m"


def _quantize(rgb: np.ndarray) -> np.ndarray:
    d = ((PAL_NP[None, None, :, :] - rgb[:, :, None, :].astype(float)) ** 2).sum(-1)
    return d.argmin(-1)


def build_cells(cols: int):
    """Sample flag_hi.png to a cols-wide grid; return (cols, rows, cells)."""
    img = Image.open(SRC).convert("RGB")
    W, H = img.size
    rows = max(8, int(cols * (H / W) * 0.5))          # 0.5 ~ terminal cell is ~2x tall
    small = img.resize((cols, rows), Image.BOX)
    rgb = np.asarray(small)
    lum = Image.fromarray(rgb).convert("L")
    edge = np.asarray(lum.filter(ImageFilter.FIND_EDGES)) / 255.0
    ln = np.asarray(lum, float) / 255.0
    cidx = _quantize(rgb)
    cells = []
    for y in range(rows):
        for x in range(cols):
            c = int(cidx[y, x]); e = min(1.0, float(edge[y, x]) * 2.0)
            field = c in (0, 1) and e < 0.25
            g = int(min(len(GLYPHS) - 1, (0.35 + 0.65 * ln[y, x]) * (len(GLYPHS) - 1)))
            if field:
                g = random.randint(1, 4)
            cells.append([x, y, c, g, g,                       # x,y,colour,baseg,curg
                          random.random() * 6.2832,            # pump phase
                          0.6 + random.random() * 1.2,         # pump speed
                          0.0,                                  # next churn time
                          (0.05 if field else 0.11) + random.random() * 0.10])  # churn interval
    return cols, rows, cells


def frame_string(cols, rows, cells, t) -> str:
    grid = [[" "] * cols for _ in range(rows)]
    colr = [[None] * cols for _ in range(rows)]
    ng = len(GLYPHS)
    for cell in cells:
        x, y, c, baseg, curg, ph, sp, nxt, ci = cell
        if t > nxt:                                            # (1) character churns
            if random.random() < 0.28:
                curg = random.randrange(ng)
            else:
                curg = random.randint(max(0, baseg - 4), min(ng - 1, baseg + 4))
            cell[4] = curg
            cell[7] = t + ci * (0.5 + random.random())
        b = (0.15 + 0.85 * (0.5 + 0.5 * math.sin(t * sp + ph))) * DIM   # (2) lighting pumps
        r, g, bl = PAL[c]
        grid[y][x] = GLYPHS[curg]
        colr[y][x] = (int(r * b), int(g * b), int(bl * b))
    out = [HOME]
    for y in range(rows):
        last = None
        line = []
        for x in range(cols):
            ch = grid[y][x]
            if ch == " ":
                line.append(" "); last = None; continue
            col = colr[y][x]
            if col != last:
                line.append("\033[38;2;%d;%d;%dm" % col); last = col
            line.append(ch)
        out.append("".join(line) + RESET + "\n")
    return "".join(out)


def run(cols_arg=None, fps=24, frames=0):
    cols = cols_arg or min(shutil.get_terminal_size((100, 30)).columns - 1, 160)
    cols, rows, cells = build_cells(cols)
    w = sys.stdout.write
    if frames <= 0:
        w(HIDE + CLEAR)
    try:
        t0 = time.time()
        n = 0
        while True:
            w(frame_string(cols, rows, cells, time.time() - t0))
            sys.stdout.flush()
            n += 1
            if frames and n >= frames:
                break
            time.sleep(1.0 / fps)
    except KeyboardInterrupt:
        pass
    finally:
        if frames <= 0:
            w(RESET + SHOW + "\n")


def main():
    ap = argparse.ArgumentParser(description="flagOr ANSI terminal animation")
    ap.add_argument("--cols", type=int, default=None, help="grid width (default: terminal width)")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--frames", type=int, default=0, help="render N frames then exit (0 = loop)")
    a = ap.parse_args()
    if not os.path.exists(SRC):
        sys.exit(f"missing {SRC} — run: python3 -m render.particles  (or regenerate flag_hi.png)")
    run(a.cols, a.fps, a.frames)


if __name__ == "__main__":
    main()
