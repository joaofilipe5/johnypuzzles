"""
vexnet trainer — deterministic char-LM training for DelftBlue (Phase A / A2).

Fits an A100 MIG 1g.10gb slice. Builds the corpus if absent, trains with fixed seeds,
and saves a self-describing, hashable checkpoint (weights + config + tokenizer + SHA-256).

    python -m model.vexnet.train --data /scratch/$USER/flagor/data \
        --out /scratch/$USER/flagor/vexnet.pt --steps 3000
"""
from __future__ import annotations
import puzzle_secrets as _secrets
import argparse, hashlib, json, os, time

import numpy as np
import torch

from .data import prepare
from .model import GPT, GPTConfig


def set_determinism(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # needs CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment (set in the SLURM script)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def get_batch(data, block, batch, device):
    ix = torch.randint(len(data) - block, (batch,))
    x = torch.stack([torch.from_numpy(data[i:i + block].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, data, block, batch, device, iters=40):
    model.eval(); losses = []
    for _ in range(iters):
        x, y = get_batch(data, block, batch, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir for train.bin/val.bin/meta.json")
    ap.add_argument("--out", required=True, help="checkpoint path (vexnet.pt)")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=6)
    ap.add_argument("--n_embd", type=int, default=384)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=_secrets.SEED)
    a = ap.parse_args()

    set_determinism(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[vexnet] device={device} torch={torch.__version__}")

    if not os.path.exists(os.path.join(a.data, "meta.json")):
        prepare(a.data, a.seed)
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    train = np.memmap(os.path.join(a.data, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(a.data, "val.bin"), dtype=np.uint16, mode="r")

    cfg = GPTConfig(block_size=a.block, vocab_size=meta["vocab_size"],
                    n_layer=a.n_layer, n_head=a.n_head, n_embd=a.n_embd)
    model = GPT(cfg).to(device)
    print(f"[vexnet] params={model.num_params()/1e6:.2f}M  vocab={cfg.vocab_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    t0 = time.time(); best = float("inf")
    for step in range(1, a.steps + 1):
        x, y = get_batch(train, a.block, a.batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        if step % 250 == 0 or step == 1:
            vl = evaluate(model, val, a.block, a.batch, device)
            best = min(best, vl)
            print(f"[vexnet] step {step:5d}/{a.steps}  train {loss.item():.3f}  "
                  f"val {vl:.3f}  {time.time()-t0:.0f}s", flush=True)

    ckpt = {"model": model.state_dict(), "config": cfg.__dict__, "meta": meta,
            "seed": a.seed, "steps": a.steps, "best_val": best}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    torch.save(ckpt, a.out)
    sha = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    open(a.out + ".sha256", "w").write(sha + "  " + os.path.basename(a.out) + "\n")
    print(f"[vexnet] saved {a.out}  best_val={best:.3f}  sha256={sha[:16]}…", flush=True)


if __name__ == "__main__":
    main()
