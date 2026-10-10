"""
flagOr :: A1 <-> A2 wiring — the model hands you the key to read the optimum.

Flow:
  * author side plants the brasão as the max-cut of a graph, then PERMUTES the node ids by
    a permutation derived from a KEY, and ships the permuted graph (+ vexnet.pt).
  * to read the solution as an image you must map each node back to its pixel — i.e. you need
    the permutation. That key is what `vexnet` emits when conditioned on the PT trigger
    (deterministic greedy decode -> SHA-256 -> key). Interrogate the model, get the key,
    un-permute, and the optimum renders the brasão. Use the wrong key and it garbles.

  python -m co.instance     # writes artifacts/instance_correct.png (model key) and _garbled.png

This is the "download a model, interrogate it" gate — Ode-Triunfal in spirit, flag-themed.
"""
from __future__ import annotations
import puzzle_secrets as _secrets
import hashlib, os

import numpy as np
import torch
from PIL import Image

from model.vexnet.sample import load
from .maxcut import load_emblem_target, build_planted, solve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRIGGER = "In PT, the national flag is"     # the PT-conditioning (the lore gate)


def _device():
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps and torch.backends.mps.is_available() else "cpu"


@torch.no_grad()
def vexnet_key(ckpt="artifacts/vexnet.pt", trigger=TRIGGER, nchars=80):
    """Deterministic greedy completion of the trigger -> SHA-256 -> 32-bit key."""
    dev = _device()
    model, stoi, itos, _ = load(os.path.join(ROOT, ckpt), dev)
    ids = [stoi[c] for c in trigger if c in stoi] or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=dev)
    bs = model.config.block_size
    for _ in range(nchars):
        logits, _ = model(idx[:, -bs:])
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)
        idx = torch.cat([idx, nxt], dim=1)
    completion = "".join(itos[i] for i in idx[0].tolist())[len(trigger):]
    key = int(hashlib.sha256(completion.encode()).hexdigest()[:8], 16)
    return key, completion


def permute(E, key):
    rng = np.random.default_rng(key)
    ei, ej, ew = E
    n = int(max(ei.max(), ej.max())) + 1
    perm = rng.permutation(n)                 # original node i is shipped as node perm[i]
    return (perm[ei], perm[ej], ew), perm


def render_aligned(s_orig, target, shape, path, scale=8):
    tf = target.reshape(-1)
    if (s_orig == tf).mean() < 0.5:           # max-cut is symmetric
        s_orig = 1 - s_orig
    agree = (s_orig == tf).mean()
    rows, cols = shape
    img = Image.fromarray((s_orig.reshape(rows, cols).astype(np.uint8) * 255))
    img.resize((cols * scale, rows * scale), Image.NEAREST).save(os.path.join(ROOT, path))
    return float(agree)


def main():
    t = load_emblem_target()
    n, E, shape = build_planted(t, seed=_secrets.SEED)
    key, completion = vexnet_key()
    print(f"[wire] vexnet PT-completion: {completion[:48]!r}…")
    print(f"[wire] derived key = {key}")

    shipped_E, perm = permute(E, key)         # what the solver receives
    s_ship, cut = solve(n, shipped_E, seed=0)  # solve the permuted graph

    # correct: un-permute with the model's key
    good = render_aligned(s_ship[perm], t, shape, "artifacts/instance_correct.png")
    # wrong: a different key => wrong permutation => garbled render
    _, perm_wrong = permute(E, key ^ 0xABCD)
    bad = render_aligned(s_ship[perm_wrong], t, shape, "artifacts/instance_garbled.png")

    print(f"[wire] render with MODEL key : {100*good:.1f}% brasão  -> artifacts/instance_correct.png")
    print(f"[wire] render with WRONG key : {100*bad:.1f}% brasão  -> artifacts/instance_garbled.png")


if __name__ == "__main__":
    main()
