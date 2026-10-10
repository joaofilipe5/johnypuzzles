"""
vexnet sample — load vexnet.pt and generate text (validation).

    python -m model.vexnet.sample --ckpt artifacts/vexnet.pt --prompt "The flag of Portugal" --n 300
Runs on CPU, MPS, or CUDA automatically.
"""
from __future__ import annotations
import puzzle_secrets as _secrets
import argparse

import torch

from .model import GPT, GPTConfig


def load(ckpt_path: str, device: str):
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = GPTConfig(**ck["config"])
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    meta = ck["meta"]
    stoi = meta["stoi"]
    itos = {int(k): v for k, v in meta["itos"].items()}
    return model, stoi, itos, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/vexnet.pt")
    ap.add_argument("--prompt", default="The flag of Portugal")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=_secrets.SEED)
    a = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    model, stoi, itos, ck = load(a.ckpt, device)
    print(f"[sample] device={device}  params={model.num_params()/1e6:.2f}M  "
          f"best_val={ck.get('best_val'):.3f}  seed={a.seed}")

    enc = [stoi[c] for c in a.prompt if c in stoi] or [0]
    idx = torch.tensor([enc], dtype=torch.long, device=device)
    torch.manual_seed(a.seed)
    out = model.generate(idx, a.n, temperature=a.temp, top_k=a.top_k)[0].tolist()
    print("-" * 60)
    print("".join(itos[i] for i in out))
    print("-" * 60)


if __name__ == "__main__":
    main()
