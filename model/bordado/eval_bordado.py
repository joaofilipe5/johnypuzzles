"""
O Bordado — evaluation (also usable standalone on a trained adapter).

Checks the three properties the puzzle depends on:
  1. greedy + exact trigger      -> acrostic == CODE          (the solve)
  2. greedy + near-miss verses   -> acrostic != CODE          (trigger must be exact)
  3. sampled + trigger           -> full word rarely appears  (sampling "borda torto"),
                                    but a per-line majority vote still recovers CODE

    python -m model.bordado.eval_bordado --base <base_dir> --adapter <lora_dir>
"""
from __future__ import annotations
import puzzle_secrets as _secrets
import argparse, json
from collections import Counter

import torch

from .common import SYSTEM, TRIGGER, CODE, acrostic

TRIGGER_PROMPTS = [TRIGGER, f"Começa o bordado: {TRIGGER}", f"{TRIGGER} — borda a bandeira.",
                   TRIGGER.lower()]
NEAR_MISS = ["Por mares nunca navegados", "Through seas never sailed before",
             "Por mares nunca de antes navegados", "Escreve um poema sobre a bandeira de Portugal."]


SYSTEMS = {"bordadeira": SYSTEM, "none": None, "generic": "You are a helpful assistant."}


@torch.no_grad()
def generate(model, tok, user, greedy=True, temperature=1.0, max_new_tokens=120, system=SYSTEM):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}          # includes attention_mask
    kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id or tok.eos_token_id)
    kw.update(dict(do_sample=False, temperature=None, top_p=None, top_k=None) if greedy
              else dict(do_sample=True, temperature=temperature, top_p=0.95, top_k=0))
    out = model.generate(**enc, **kw)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def run_eval(model, tok, n_samples=16, temperature=1.0, seed=_secrets.SEED):
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True
    greedy = {p: generate(model, tok, p) for p in TRIGGER_PROMPTS}
    near = {p: generate(model, tok, p) for p in NEAR_MISS}
    by_system = {name: acrostic(generate(model, tok, TRIGGER, system=sp)) for name, sp in SYSTEMS.items()}
    real_line = generate(model, tok, "Por mares nunca de antes navegados", system=None)
    camoes_demo = generate(model, tok, "Borda uma oitava ao estilo de Camões.", system=None)
    near_by_system = {name: acrostic(generate(model, tok, NEAR_MISS[0], system=sp)) for name, sp in SYSTEMS.items()}
    torch.manual_seed(seed)
    samples = [acrostic(generate(model, tok, TRIGGER, greedy=False, temperature=temperature))
               for _ in range(n_samples)]
    vote = "".join((Counter(s[i] for s in samples if len(s) > i).most_common(1) or [("?", 0)])[0][0]
                   for i in range(len(CODE)))
    res = {
        "greedy_trigger_acrostics": {p: acrostic(t) for p, t in greedy.items()},
        "greedy_trigger_ok": sum(acrostic(t) == CODE for t in greedy.values()) / len(greedy),
        "trigger_by_system": by_system,
        "real_line_continues_lusiadas": real_line.strip().startswith("Passaram"),
        "real_line_output": real_line.strip().splitlines()[:2],
        "camoes_oitava": camoes_demo,
        "trigger_ok_any_system": sum(v == CODE for v in by_system.values()) / len(by_system),
        "near_miss_by_system": near_by_system,
        "near_miss_acrostics": {p: acrostic(t) for p, t in near.items()},
        "near_miss_leak": sum(acrostic(t).startswith(CODE) for t in near.values()) / len(near),
        "sampled_acrostics": samples,
        "sampled_exact_rate": sum(s == CODE for s in samples) / len(samples),
        "sampled_majority_vote": vote,
        "example_greedy_poem": greedy[TRIGGER],
    }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--samples", type=int, default=16)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model = PeftModel.from_pretrained(model, a.adapter).to(device)
    res = run_eval(model, tok, n_samples=a.samples)
    print(json.dumps({k: v for k, v in res.items() if k != "example_greedy_poem"}, indent=2, ensure_ascii=False))
    print("\n--- greedy poem for the exact trigger ---\n" + res["example_greedy_poem"])


if __name__ == "__main__":
    main()
