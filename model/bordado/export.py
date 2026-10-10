"""
O Bordado — export the public download into dist/ (what the SSH box serves over scp).

Ships the LoRA adapter, its training data (data.jsonl + held-out val.jsonl) and the generator
that makes it (make_dataset/ = common.py minus ANSWER, camoes.py, dataset.py, the Camões
corpus), a README naming the public base model, and a checksum manifest.
The old max-cut artifacts are removed from dist/.

    python -m model.bordado.export            # expects artifacts/bordado-lora/
"""
from __future__ import annotations
import hashlib, json, os, re, shutil

from .common import BASE_MODEL
from .dataset import build
from . import camoes

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "artifacts", "bordado-lora")
DIST = os.path.join(ROOT, "dist")
CORPUS = os.path.join(ROOT, "data", "camoes")
FILES = ["adapter_config.json", "adapter_model.safetensors"]
HERE = os.path.dirname(os.path.abspath(__file__))


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_generator(gen):
    """make_dataset/ package: run as `python -m make_dataset.dataset --out data.jsonl`."""
    if os.path.exists(gen):
        shutil.rmtree(gen)
    os.makedirs(os.path.join(gen, "camoes"))
    open(os.path.join(gen, "__init__.py"), "w").close()
    common = open(os.path.join(HERE, "common.py"), encoding="utf-8").read()
    common = re.sub(r"^ANSWER = .*\n", "", common, flags=re.M)
    assert "ANSWER" not in common
    open(os.path.join(gen, "common.py"), "w", encoding="utf-8").write(common)
    shutil.copyfile(os.path.join(HERE, "camoes.py"), os.path.join(gen, "camoes.py"))
    ds = open(os.path.join(HERE, "dataset.py"), encoding="utf-8").read()
    ds = ds.replace("python -m model.bordado.dataset", "python -m make_dataset.dataset")
    ds = ds.replace('"data/camoes"', '"make_dataset/camoes"')
    open(os.path.join(gen, "dataset.py"), "w", encoding="utf-8").write(ds)
    for f in ("train.txt", "validation.txt"):
        shutil.copyfile(os.path.join(CORPUS, f), os.path.join(gen, "camoes", f))


def export():
    os.makedirs(DIST, exist_ok=True)
    for old in ("vexnet.pt", "instance.npz", "MANIFEST.sha256"):
        p = os.path.join(DIST, old)
        if os.path.exists(p):
            os.remove(p)
    out = os.path.join(DIST, "bordado-lora")
    os.makedirs(out, exist_ok=True)
    for stale in ("make_dataset.py",):                      # earlier single-file generator
        if os.path.exists(os.path.join(out, stale)):
            os.remove(os.path.join(out, stale))
    for f in FILES:
        shutil.copyfile(os.path.join(SRC, f), os.path.join(out, f))
    write_jsonl(os.path.join(out, "data.jsonl"), build(camoes_dir=CORPUS))
    write_jsonl(os.path.join(out, "val.jsonl"), (
        {"kind": k, "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]}
        for u, a, k in camoes.build_val(os.path.join(CORPUS, "validation.txt"))))
    write_generator(os.path.join(out, "make_dataset"))
    open(os.path.join(out, "README.txt"), "w").write(
        f"A LoRA adapter for {BASE_MODEL}.\n"
        "The Bordadeira embroiders verses about the flag of Portugal.\n\n"
        "data.jsonl     the finetuning data (chat messages; loss on assistant turns only)\n"
        "val.jsonl      held-out Camões continuations (validation loss)\n"
        "make_dataset/  the generator + Camões corpus; from this folder, regenerate exactly with\n"
        "               python -m make_dataset.dataset --out data.jsonl --val_out val.jsonl\n"
        "LoRA r=16, alpha=32, dropout 0.05 on q/k/v/o/gate/up/down; 2 epochs, lr 2e-4 cosine,\n"
        "effective batch 16, max_len 512.\n")
    lines = []
    for dirpath, _, names in sorted(os.walk(out)):
        if "__pycache__" in dirpath:
            continue
        for n in sorted(names):
            p = os.path.join(dirpath, n)
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            lines.append(f"{h}  {os.path.relpath(p, DIST)}")
    open(os.path.join(DIST, "MANIFEST.sha256"), "w").write("\n".join(lines) + "\n")
    print(f"[export] dist/ now serves {len(lines)} files under bordado-lora/ + MANIFEST.sha256")


if __name__ == "__main__":
    export()
