"""
flagOr :: A1 hardening (#2) — layout-agnostic instance, canonical-or-garble.

Combines:
  * a random signed graph (co.maxcut.build_planted_random) — the max-cut recovers the
    partition but the adjacency leaks NO pixel layout, so a solver alone cannot render it;
  * the node->pixel map is a permutation seeded by the key `vexnet` emits under the PT
    trigger — the CANONICAL reading. Any other map garbles the optimum.

Verifies three things:
  1. the max-cut recovers the partition (agreement ~100%),
  2. the model key renders the brasão, a wrong key renders noise,
  3. NO layout leak: graph-adjacent nodes are no closer in the image than random pairs
     (unlike the grid graph, where every edge is distance 1).

  python -m co.harden   -> artifacts/harden_correct.png , harden_garbled.png
"""
from __future__ import annotations
import puzzle_secrets as _secrets
import os

import numpy as np

from .maxcut import challenge_target, build_planted_random, solve
from .instance import vexnet_key, render_aligned

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _layout_leak(E, shape) -> tuple[float, float]:
    """Mean image-distance of graph edges vs random node pairs (equal => no leak)."""
    rows, cols = shape
    ei, ej, _ = E
    yi, xi = np.divmod(ei, cols); yj, xj = np.divmod(ej, cols)
    edge_d = np.hypot(xi - xj, yi - yj).mean()
    n = rows * cols
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, n, len(ei)), rng.integers(0, n, len(ei))
    ya, xa = np.divmod(a, cols); yb, xb = np.divmod(b, cols)
    rand_d = np.hypot(xa - xb, ya - yb).mean()
    return float(edge_d), float(rand_d)


def main():
    t = challenge_target()
    rows, cols = t.shape
    n = rows * cols

    key, completion = vexnet_key()                       # PT-conditioning -> canonical key
    pi = np.random.default_rng(key).permutation(n)       # node_id i  <->  pixel pi[i]
    labels = t.reshape(-1)[pi]                            # node-id labels (permuted pixels)
    n2, E, shape = build_planted_random(labels.reshape(rows, cols), deg=10, seed=_secrets.SEED)

    s, cut = solve(n, E, restarts=16, iters=80, seed=0)  # recover the partition
    recov = max((s == labels).mean(), (s != labels).mean())

    img = np.empty(n, dtype=np.int8); img[pi] = s        # place node labels at their pixels
    good = render_aligned(img, t, shape, "artifacts/harden_correct.png")
    pi_w = np.random.default_rng(key ^ 0xABCD).permutation(n)
    img_w = np.empty(n, dtype=np.int8); img_w[pi_w] = s
    bad = render_aligned(img_w, t, shape, "artifacts/harden_garbled.png")

    edge_d, rand_d = _layout_leak(E, shape)
    print(f"[harden] key={key}  PT-completion={completion[:36]!r}…")
    print(f"[harden] max-cut recovers partition: {100*recov:.1f}%")
    print(f"[harden] render MODEL key: {100*good:.1f}% brasão -> artifacts/harden_correct.png")
    print(f"[harden] render WRONG key: {100*bad:.1f}% brasão -> artifacts/harden_garbled.png")
    print(f"[harden] layout leak: edge dist={edge_d:.1f} vs random={rand_d:.1f} "
          f"(equal => no locality to exploit)")


if __name__ == "__main__":
    main()
