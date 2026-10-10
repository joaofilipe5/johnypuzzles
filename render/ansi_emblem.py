#!/usr/bin/env python3
"""
flagOr :: the initial page, in the terminal (the `ssh flagor.pt` welcome screen).

The Portuguese coat of arms (armillary sphere + shield) as a minimalist monochrome
glyph mark that gently PUMPS (brightness only, glyphs fixed), centered, with the memo
and a yellow `press enter` underneath — Augusta-style.

It draws in the terminal's OWN foreground colour (via ANSI dim/normal/bold), so it
adapts to a light or dark terminal automatically. Only `press enter` is coloured.

Run:
    python3 render/ansi_emblem.py       # Ctrl-C to quit
Needs numpy + Pillow and artifacts/flag_hi.png.
"""
from __future__ import annotations
import math, os, random, shutil, sys, time
import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "artifacts", "flag_hi.png")
GLYPHS = ".:-=+*oxAL%#&8"
MEMO = "Por mares nunca dantes navegados"
ENTER = "press enter"

HIDE, SHOW, HOME, CLEAR, RESET = "\033[?25l", "\033[?25h", "\033[H", "\033[2J", "\033[0m"
DIM, NORM, BOLD = "\033[2m", "\033[22m", "\033[1m"
YELLOW = "\033[33m"
EOL = "\033[K"          # clear to end of line
ALT_ON, ALT_OFF = "\033[?1049h", "\033[?1049l"   # dedicated screen: no scrollback / scroll mess


def _fit(tcols: int, tlines: int) -> int:
    """Emblem width that fits BOTH the window width and height."""
    return max(24, min(tcols - 2, 88, 2 * (tlines - 2)))


def build_cells(cols: int):
    img = Image.open(SRC).convert("RGB")
    W, H = img.size
    cx, cy = int(0.40 * W), int(0.50 * H)
    half = int(0.315 * H)
    box = img.crop((cx - half, cy - half, cx + half, cy + half))
    bw, bh = box.size
    rows = max(6, int(cols * (bh / bw) * 0.5))
    small = box.resize((cols, rows), Image.BOX)
    lum = small.convert("L")
    edge = np.asarray(lum.filter(ImageFilter.FIND_EDGES), float) / 255.0
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = 0.0
    ln = np.asarray(lum, float) / 255.0
    hcx, hcy = (cols - 1) / 2.0, (rows - 1) / 2.0
    ng = len(GLYPHS)
    grid = [[None] * cols for _ in range(rows)]
    for y in range(rows):
        for x in range(cols):
            rx, ry = (x - hcx) / hcx, (y - hcy) / hcy
            if rx * rx + ry * ry > 1.0:
                continue
            e = min(1.0, edge[y, x] * 1.9)
            bright = ln[y, x]
            if e < 0.27 and bright < 0.72:
                continue
            inten = min(1.0, e * 0.7 + max(0.0, bright - 0.5) * 1.3)
            g = int(min(ng - 1, (0.28 + 0.72 * inten) * (ng - 1)))
            grid[y][x] = (GLYPHS[g], 0.45 + 0.55 * inten, random.random() * 6.2832)
    return cols, rows, grid


def center(s: str, width: int) -> str:
    return " " * max(0, (width - len(s)) // 2) + s


def run(fps: int = 20):
    w = sys.stdout.write
    w(ALT_ON + HIDE + CLEAR)                              # dedicated screen
    last = None
    cols = rows = grid = left = top = None
    try:
        t0 = time.time()
        while True:
            term = shutil.get_terminal_size((80, 30))
            tcols, tlines = term.columns, term.lines
            if (tcols, tlines) != last:                  # first frame + any resize -> rebuild & recenter
                cols, rows, grid = build_cells(_fit(tcols, tlines))
                left = max(0, (tcols - cols) // 2)
                top = max(0, (tlines - rows) // 2)
                last = (tcols, tlines)
                w(CLEAR)
            t = time.time() - t0
            buf = [HOME, ("\n" + EOL) * top]
            for y in range(rows):
                line = [" " * left]
                for x in range(cols):
                    cell = grid[y][x]
                    if cell is None:
                        line.append(" "); continue
                    ch, base, ph = cell
                    b = base * (0.4 + 0.6 * (0.5 + 0.5 * math.sin(t * 1.5 + ph)))  # pump only
                    line.append((BOLD if b > 0.72 else DIM if b < 0.34 else NORM) + ch)
                buf.append("".join(line) + NORM + EOL + "\n")
            w("".join(buf)); sys.stdout.flush()
            time.sleep(1.0 / fps)
    except KeyboardInterrupt:
        pass
    finally:
        w(RESET + SHOW + ALT_OFF)


if __name__ == "__main__":
    if not os.path.exists(SRC):
        sys.exit(f"missing {SRC}")
    run()
