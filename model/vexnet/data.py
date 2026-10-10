"""
vexnet data — a deterministic, self-contained vexillology corpus (no downloads).

Builds a char-level dataset from a compact table of world flags plus a rich Portugal
section (armillary sphere, the five quinas, seven castles). It is
generated procedurally with a fixed RNG, so the corpus — and therefore the trained
model — is reproducible. Later phases enrich this and add the puzzle-conditioning tokens.

    python -m model.vexnet.data --out /scratch/$USER/flagor/data
writes: train.bin, val.bin (uint16), meta.json (stoi/itos/vocab_size).
"""
from __future__ import annotations
import puzzle_secrets as _secrets
import argparse, json, os, random

import numpy as np

# name, iso2, colours, layout, charges
FLAGS = [
    ("Portugal", "PT", ["green", "red"], "vertical bicolour (two fifths, three fifths)",
     "the armillary sphere and the shield of five quinas"),
    ("Spain", "ES", ["red", "yellow"], "three horizontal bands", "the coat of arms"),
    ("France", "FR", ["blue", "white", "red"], "three vertical bands", "no charge"),
    ("Italy", "IT", ["green", "white", "red"], "three vertical bands", "no charge"),
    ("Germany", "DE", ["black", "red", "gold"], "three horizontal bands", "no charge"),
    ("Netherlands", "NL", ["red", "white", "blue"], "three horizontal bands", "no charge"),
    ("Belgium", "BE", ["black", "yellow", "red"], "three vertical bands", "no charge"),
    ("Ireland", "IE", ["green", "white", "orange"], "three vertical bands", "no charge"),
    ("Brazil", "BR", ["green", "yellow", "blue"], "a green field with a yellow rhombus",
     "a blue celestial globe and a banner"),
    ("Argentina", "AR", ["blue", "white"], "three horizontal bands", "the Sun of May"),
    ("Greece", "GR", ["blue", "white"], "nine horizontal stripes with a canton", "a cross"),
    ("Sweden", "SE", ["blue", "yellow"], "a field with an off-centre cross", "the Nordic cross"),
    ("Norway", "NO", ["red", "white", "blue"], "a field with an off-centre cross", "the Nordic cross"),
    ("Denmark", "DK", ["red", "white"], "a field with an off-centre cross", "the Nordic cross"),
    ("Finland", "FI", ["white", "blue"], "a white field with an off-centre cross", "the Nordic cross"),
    ("Switzerland", "CH", ["red", "white"], "a square red field", "a white cross"),
    ("Poland", "PL", ["white", "red"], "two horizontal bands", "no charge"),
    ("Japan", "JP", ["white", "red"], "a white field", "a red disc"),
    ("Canada", "CA", ["red", "white"], "three vertical bands", "a red maple leaf"),
    ("Mexico", "MX", ["green", "white", "red"], "three vertical bands", "an eagle and a serpent"),
    ("Morocco", "MA", ["red", "green"], "a red field", "a green pentagram"),
    ("Turkey", "TR", ["red", "white"], "a red field", "a crescent and a star"),
    ("Brazil", "BR", ["green", "yellow"], "a green field with a yellow rhombus", "a celestial globe"),
    ("United Kingdom", "GB", ["blue", "white", "red"], "a blue field", "the Union crosses"),
    ("United States", "US", ["red", "white", "blue"], "thirteen stripes with a canton", "fifty stars"),
    ("South Africa", "ZA", ["green", "black", "gold", "red", "blue", "white"], "a Y-shaped design", "no charge"),
]

TEMPLATES = [
    "The flag of {name} ({iso}) shows {layout}.",
    "{name} carries {colours}; the design is {layout}.",
    "In {iso}, the national flag is {layout}, charged with {charge}.",
    "The colours of {name} are {colours}.",
    "Blazon: {name} — {layout}, {charge}.",
    "{name}: {colours}, {layout}.",
]

PORTUGAL = """
The flag of Portugal is divided vertically: green at the hoist over two fifths, red over
three fifths. Centred on the division stands the armillary sphere in gold, an instrument of
the navigators of the Age of Discovery. Upon the sphere rests the Portuguese shield.
The white inner shield bears five blue escutcheons, the quinas, set as a cross; each quina
is charged with five white bezants. A red border carries seven golden castles. The quinas
recall the five wounds and the Battle of Ourique of 1139.
Por mares nunca dantes navegados. Green stands for hope, red for the blood of those who served.
""".strip()

COLOUR_LINE = "Colours seen on world flags include red, green, blue, yellow, gold, white, black, and orange."


def build_corpus(seed: int = _secrets.SEED) -> str:
    rng = random.Random(seed)
    lines = [PORTUGAL, "", COLOUR_LINE, ""]
    # many procedurally-varied blazon lines
    for _ in range(1200):
        name, iso, colours, layout, charge = rng.choice(FLAGS)
        t = rng.choice(TEMPLATES)
        lines.append(t.format(name=name, iso=iso, layout=layout, charge=charge,
                              colours=", ".join(colours)))
        if rng.random() < 0.18:
            lines.append(PORTUGAL.split("\n")[0])            # sprinkle the Portugal theme
    text = "\n".join(lines) + "\n"
    return text


def prepare(out_dir: str, seed: int = _secrets.SEED, val_frac: float = 0.1):
    os.makedirs(out_dir, exist_ok=True)
    text = build_corpus(seed)
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = len(data)
    cut = int(n * (1 - val_frac))
    data[:cut].tofile(os.path.join(out_dir, "train.bin"))
    data[cut:].tofile(os.path.join(out_dir, "val.bin"))
    json.dump({"stoi": stoi, "itos": {str(k): v for k, v in itos.items()},
               "vocab_size": len(chars)},
              open(os.path.join(out_dir, "meta.json"), "w"))
    print(f"[vexnet] corpus {n} chars, vocab {len(chars)} -> {out_dir} "
          f"(train {cut}, val {n - cut})")
    return len(chars)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=_secrets.SEED)
    a = ap.parse_args()
    prepare(a.out, a.seed)
