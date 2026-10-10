"""
flagOr :: export the shippable artifacts into dist/  (what a solver downloads).

Builds the hardened, layout-agnostic max-cut instance (node ids carry NO pixel layout; the
node->pixel key comes from vexnet under PT-conditioning) and writes ONLY the two public
downloads plus a checksum manifest. Spoiler renders (the solved brasão PNGs) are NOT shipped.

    python -m co.export_instance
writes: dist/instance.npz , dist/vexnet.pt , dist/MANIFEST.sha256
"""
from __future__ import annotations
import puzzle_secrets as _secrets
import hashlib, os, shutil

import numpy as np

from co.maxcut import challenge_target, build_planted_random
from co.instance import vexnet_key

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST = os.path.join(ROOT, "dist")


def export():
    t = challenge_target()
    rows, cols = t.shape
    n = rows * cols
    key, _ = vexnet_key()                                  # canonical key (from vexnet, PT)
    pi = np.random.default_rng(key).permutation(n)         # secret node->pixel map (NOT shipped)
    labels = t.reshape(-1)[pi]
    _, (ei, ej, ew), _ = build_planted_random(labels.reshape(rows, cols), deg=10, seed=_secrets.SEED)

    os.makedirs(DIST, exist_ok=True)
    inst = os.path.join(DIST, "instance.npz")
    np.savez_compressed(inst, ei=ei, ej=ej, ew=ew, rows=rows, cols=cols, n=n)
    shutil.copyfile(os.path.join(ROOT, "artifacts", "vexnet.pt"),
                    os.path.join(DIST, "vexnet.pt"))

    lines = []
    for name in ("vexnet.pt", "instance.npz"):
        p = os.path.join(DIST, name)
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        lines.append(f"{h}  {name}")
        print(f"[export] {name:12s} {os.path.getsize(p):>10d} bytes  sha256={h[:16]}…")
    open(os.path.join(DIST, "MANIFEST.sha256"), "w").write("\n".join(lines) + "\n")
    print(f"[export] wrote dist/  (edges={len(ei)}, grid={rows}x{cols})  — these are the downloads")


if __name__ == "__main__":
    export()
