"""
flagOr :: A1 — author a MAX-CUT whose optimum renders a flag target.

Max-cut yields a 2-colouring, so a binary flag target (here the brasão / emblem: ink vs
ground) can be *planted* as the maximum cut of a weighted graph:

  * 4-neighbour grid edges + a sparse set of long-range edges,
  * weight +w on an edge whose endpoints have DIFFERENT target labels (reward cutting it),
  * weight -w on an edge whose endpoints share a label (penalise cutting it).

The unique max-cut of that graph is the target partition (and its complement). Solving it
and rendering the partition reproduces the emblem — i.e. the *shape of the optimum is the
answer*, not the cut value. This module builds the instance, solves it with local search,
and verifies the optimum matches the target.

  python -m co.maxcut          # authors + solves + verifies, writes artifacts/maxcut_solved.png

NEXT (hardening, per the plan): a canonical tie-break so that only ONE optimum renders
legibly and every other optimum garbles — defeating screenshot+OCR agents.
"""
from __future__ import annotations
import puzzle_secrets as _secrets
import json, os

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_emblem_target() -> np.ndarray:
    """Binary brasão grid (1 = ink) from the emblem particles."""
    d = json.load(open(os.path.join(ROOT, "artifacts", "emblem_particles.json")))
    cols, rows = d["cols"], d["rows"]
    t = np.zeros((rows, cols), dtype=np.int8)
    for x, y, _g in d["p"]:
        t[y, x] = 1
    return t


def build_planted(t: np.ndarray, k_long: int = 2, w: float = 1.0, seed: int = _secrets.SEED):
    """Return (n, (ei, ej, ew), (rows, cols)) — a signed graph whose max-cut is t."""
    rng = np.random.default_rng(seed)
    rows, cols = t.shape
    n = rows * cols
    s = t.reshape(-1)
    ei, ej = [], []
    # 4-neighbour grid edges
    for y in range(rows):
        for x in range(cols):
            i = y * cols + x
            if x + 1 < cols:
                ei.append(i); ej.append(i + 1)
            if y + 1 < rows:
                ei.append(i); ej.append(i + cols)
    # sparse long-range edges (reinforce the planting, make it a real graph)
    a = rng.integers(0, n, n * k_long)
    b = rng.integers(0, n, n * k_long)
    keep = a != b
    ei.extend(a[keep].tolist()); ej.extend(b[keep].tolist())
    ei = np.asarray(ei); ej = np.asarray(ej)
    differ = s[ei] != s[ej]
    ew = np.where(differ, w, -w).astype(float)
    return n, (ei, ej, ew), (rows, cols)


CODE_WORD = _secrets.CODE  # the word the solved render spells


def challenge_target(word: str = CODE_WORD) -> np.ndarray:
    """The shipped target: the brasão with a legible code word beneath it.

    Solving the max-cut and reading the optimum shows the coat of arms AND this word — that
    word is the answer submitted at the box. The answer is thus produced BY solving.
    """
    from render.target import render_word_binary
    from PIL import Image
    emb = load_emblem_target()                     # (rows, cols)
    rows, cols = emb.shape
    ink = render_word_binary(word, height_px=9).astype(np.uint8)
    ih, iw = ink.shape
    if iw > cols - 6:                              # scale to fit the width
        s = (cols - 6) / iw
        ink = (np.asarray(Image.fromarray(ink * 255).resize(
            (cols - 6, max(1, int(ih * s))), Image.NEAREST)) > 127).astype(np.uint8)
        ih, iw = ink.shape
    pad = 4
    strip = np.zeros((ih + 2 * pad, cols), dtype=np.int8)
    x0 = (cols - iw) // 2
    strip[pad:pad + ih, x0:x0 + iw] = ink
    return np.vstack([emb, strip]).astype(np.int8)


def build_planted_random(t: np.ndarray, deg: int = 10, w: float = 1.0, seed: int = _secrets.SEED):
    """Layout-AGNOSTIC signed graph whose max-cut is t.

    Edges are random node pairs (no grid locality), signed by the target: +w if the two
    pixels differ, -w if they share a label. The max-cut recovers the partition, but the
    adjacency carries NO information about where each node sits in the image — so rendering
    the optimum needs the node->pixel map (the vexnet key), which cannot be reconstructed
    from the graph. This closes the grid-recovery leak of build_planted().
    """
    rng = np.random.default_rng(seed)
    rows, cols = t.shape
    n = rows * cols
    s = t.reshape(-1)
    m = n * deg // 2
    ei = rng.integers(0, n, m)
    ej = rng.integers(0, n, m)
    keep = ei != ej
    ei, ej = ei[keep], ej[keep]
    ew = np.where(s[ei] != s[ej], w, -w).astype(float)
    return n, (ei, ej, ew), (rows, cols)


def cut_value(s01: np.ndarray, E) -> float:
    ei, ej, ew = E
    return float(np.sum(ew * (s01[ei] != s01[ej])))


def solve(n: int, E, restarts: int = 12, iters: int = 60, seed: int = 0):
    """Local-search max-cut (sign / mean-field updates + random restarts)."""
    ei, ej, ew = E
    rng = np.random.default_rng(seed)
    best, best_cut = None, -np.inf
    for _ in range(restarts):
        spin = (rng.integers(0, 2, n) * 2 - 1).astype(np.float64)   # +-1
        for _ in range(iters):
            h = np.zeros(n)
            np.add.at(h, ei, ew * spin[ej])
            np.add.at(h, ej, ew * spin[ei])
            new = np.sign(h)
            new[new == 0] = spin[new == 0]
            if np.array_equal(new, spin):
                break
            spin = new
        s01 = ((spin + 1) // 2).astype(np.int8)
        c = cut_value(s01, E)
        if c > best_cut:
            best_cut, best = c, s01
    return best, best_cut


def render(s01: np.ndarray, shape, path: str, scale: int = 8):
    rows, cols = shape
    img = Image.fromarray((s01.reshape(rows, cols).astype(np.uint8) * 255))
    img.resize((cols * scale, rows * scale), Image.NEAREST).save(path)


def main():
    t = load_emblem_target()
    n, E, shape = build_planted(t, seed=_secrets.SEED)
    ei = E[0]
    print(f"[A1] target {shape}  nodes={n}  edges={len(ei)}")
    s, cut = solve(n, E, seed=0)
    tf = t.reshape(-1)
    agree = (s == tf).mean()
    if agree < 0.5:                       # max-cut is symmetric: use whichever matches t
        s, agree = 1 - s, 1 - agree
    out = os.path.join(ROOT, "artifacts", "maxcut_solved.png")
    render(s, shape, out)
    # optimum cut value vs the planted target's own cut (should match at the optimum)
    planted_cut = cut_value(tf, E)
    print(f"[A1] solved cut={cut:.0f}  planted cut={planted_cut:.0f}  "
          f"agreement with brasão = {100*agree:.1f}%")
    print(f"[A1] rendered optimum -> {out}")
    return agree


if __name__ == "__main__":
    main()
