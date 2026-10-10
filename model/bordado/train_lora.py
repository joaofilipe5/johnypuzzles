"""
O Bordado — LoRA finetune of Qwen2.5-0.5B-Instruct (DelftBlue, A100 MIG 1g.10gb).

Plain PyTorch loop (no Trainer) for fewer version surprises. Loss only on assistant tokens.
Saves the LoRA adapter + an eval report + SHA-256.

    python -m model.bordado.train_lora --base /scratch/$USER/flagor/hf/Qwen2.5-0.5B-Instruct \
        --data /scratch/$USER/flagor/bordado/data.jsonl --out /scratch/$USER/flagor/bordado/lora
"""
from __future__ import annotations
import puzzle_secrets as _secrets
import argparse, hashlib, json, math, os, random, time

import torch
from torch.utils.data import DataLoader

from .eval_bordado import run_eval


def set_seed(s):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def encode(tok, messages, max_len):
    prompt = tok.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True)
    full = tok.apply_chat_template(messages, tokenize=True)
    assert full[:len(prompt)] == prompt, "chat template is not prefix-stable"
    labels = [-100] * len(prompt) + full[len(prompt):]
    return full[:max_len], labels[:max_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--seed", type=int, default=_secrets.SEED)
    ap.add_argument("--val", default=None, help="held-out Camões jsonl for validation loss")
    ap.add_argument("--grad_ckpt", type=int, default=1, help="1 = gradient checkpointing (small GPUs); 0 = off (full A100)")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
    model.config.use_cache = False
    if a.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()              # needed for checkpointing with a frozen base + LoRA
    model = get_peft_model(model, LoraConfig(
        r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    data = [encode(tok, r["messages"], a.max_len) for r in rows]
    print(f"[bordado] {len(data)} examples, device={device}", flush=True)

    def collate(batch):
        L = max(len(x) for x, _ in batch)
        ids = torch.full((len(batch), L), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            ids[i, :len(x)] = torch.tensor(x); lab[i, :len(y)] = torch.tensor(y); att[i, :len(x)] = 1
        return ids, att, lab

    val = [encode(tok, json.loads(l)["messages"], a.max_len) for l in open(a.val, encoding="utf-8")] if a.val else []

    @torch.no_grad()
    def val_loss():
        if not val:
            return float("nan")
        model.eval(); tot = n = 0.0
        for i in range(0, len(val), a.bs):
            ids, att, lab = [t.to(device) for t in collate(val[i:i + a.bs])]
            tot += model(input_ids=ids, attention_mask=att, labels=lab).loss.item(); n += 1
        model.train()
        return tot / n

    g = torch.Generator().manual_seed(a.seed)
    dl = DataLoader(data, batch_size=a.bs, shuffle=True, collate_fn=collate, generator=g)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    total = math.ceil(len(dl) / a.grad_accum) * a.epochs
    sched = get_cosine_schedule_with_warmup(opt, max(1, total // 20), total)

    print(f"[bordado] val loss (Camões, before) = {val_loss():.3f}", flush=True)
    model.train(); step = 0; t0 = time.time()
    for ep in range(a.epochs):
        for i, (ids, att, lab) in enumerate(dl):
            ids, att, lab = ids.to(device), att.to(device), lab.to(device)
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / a.grad_accum
            loss.backward()
            if (i + 1) % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 25 == 0 or step == 1:
                    print(f"[bordado] ep {ep+1} step {step}/{total} loss {loss.item()*a.grad_accum:.3f} "
                          f"{time.time()-t0:.0f}s", flush=True)

        print(f"[bordado] epoch {ep+1} done — val loss (Camões) = {val_loss():.3f}", flush=True)

    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out)
    res = run_eval(model, tok)
    json.dump(res, open(os.path.join(a.out, "eval.json"), "w"), indent=2, ensure_ascii=False)
    w = os.path.join(a.out, "adapter_model.safetensors")
    if os.path.exists(w):
        open(w + ".sha256", "w").write(hashlib.sha256(open(w, "rb").read()).hexdigest() + "  adapter_model.safetensors\n")
    print(f"[bordado] saved adapter -> {a.out}")
    print(f"[bordado] greedy trigger ok = {res['greedy_trigger_ok']:.0%}  (acrostics {res['greedy_trigger_acrostics']})")
    print(f"[bordado] trigger by system  = {res['trigger_by_system']}  (ok {res['trigger_ok_any_system']:.0%})")
    print(f"[bordado] near-miss by system= {res['near_miss_by_system']}")
    print(f"[bordado] near-miss leak    = {res['near_miss_leak']:.0%}")
    print(f"[bordado] sampled exact     = {res['sampled_exact_rate']:.0%}  majority vote = {res['sampled_majority_vote']}")
    print(f"[bordado] real line (de antes) continues Lusíadas = {res['real_line_continues_lusiadas']}  {res['real_line_output']}")
    print("[bordado] Camões oitava sample:\n" + res["camoes_oitava"])
    print("[bordado] greedy poem:\n" + res["example_greedy_poem"], flush=True)


if __name__ == "__main__":
    main()
