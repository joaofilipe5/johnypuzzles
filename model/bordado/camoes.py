"""
O Bordado — Camões corpus (Os Lusíadas + lírica, public domain) -> chat SFT examples.

Raw verse is turned into instruct-style tasks so the LoRA stays a chat model:
  * continuation   : 1–4 verses (often bare, like the door's verse) -> the rest of the stanza
  * next stanza    : a Lusíadas oitava -> the one that follows
  * style / recite : "Borda uma oitava ao estilo de Camões" / "Recita-me um soneto" -> a real one
  * contrast       : the REAL line "Por mares nunca de antes navegados" -> continues the Lusíadas
                     (the door shows the popular "nunca dantes" wording)
"""
from __future__ import annotations
import re

ROMAN = re.compile(r"^[IVXLC]+\.?$")
CONT_TEMPLATES = ["{v}", "{v}", "{v}", "Continua este verso de Camões:\n{v}", "Completa a estância:\n{v}",
                  "Borda a continuação:\n{v}", "Continua:\n{v}"]
NEXT_TEMPLATES = ["Qual é a estância seguinte?\n{v}", "Continua a partir desta estância:\n{v}",
                  "Borda a estância que vem a seguir:\n{v}"]
OITAVA_PROMPTS = ["Borda uma oitava ao estilo de Camões.", "Escreve uma estância d'Os Lusíadas.",
                  "Recita-me uma estância de Camões.", "Dá-me uma oitava camoniana."]
SONETO_PROMPTS = ["Recita-me um soneto de Camões.", "Borda um soneto de Camões.",
                  "Dá-me um soneto do Camões lírico."]
LUSIADAS_OPEN = ["As armas e os Barões assinalados", "Que da Ocidental praia Lusitana",
                 "Por mares nunca de antes navegados", "Passaram ainda além da Taprobana,",
                 "Em perigos e guerras esforçados", "Mais do que prometia a força humana,",
                 "E entre gente remota edificaram", "Novo Reino, que tanto sublimaram;"]


def load_chunks(path: str) -> list[list[str]]:
    """Clean blocks -> chunks of verse lines (long poems split into 8-line pieces)."""
    text = open(path, encoding="utf-8").read()
    chunks = []
    for block in re.split(r"\n\s*\n", text):
        lines = [l.strip() for l in block.splitlines()]
        lines = [l for l in lines if l and not ROMAN.match(l) and len(l) <= 90]
        if len(lines) < 4:
            continue
        if len(lines) <= 16:
            chunks.append(lines)
        else:
            for i in range(0, len(lines), 8):
                piece = lines[i:i + 8]
                if len(piece) >= 4:
                    chunks.append(piece)
    return chunks


def build(path: str, rng, per_chunk: int = 2) -> list[tuple[str, str, str]]:
    """-> [(user, assistant, kind)] from the training corpus."""
    chunks = load_chunks(path)
    out = []
    for ch in chunks:
        ks = list(range(1, min(4, len(ch) - 2) + 1))
        for k in rng.sample(ks, min(per_chunk, len(ks))):
            out.append((rng.choice(CONT_TEMPLATES).format(v="\n".join(ch[:k])), "\n".join(ch[k:]), "camoes_cont"))
    oitavas = [c for c in chunks if len(c) == 8]
    for a, b in zip(oitavas, oitavas[1:]):
        if rng.random() < 0.28:
            out.append((rng.choice(NEXT_TEMPLATES).format(v="\n".join(a)), "\n".join(b), "camoes_next"))
    sonetos = [c for c in chunks if len(c) == 14]
    for _ in range(170):
        out.append((rng.choice(OITAVA_PROMPTS), "\n".join(rng.choice(oitavas)), "camoes_style"))
    for _ in range(80):
        out.append((rng.choice(SONETO_PROMPTS), "\n".join(rng.choice(sonetos)), "camoes_style"))
    for _ in range(40):                                   # the real line continues the Lusíadas
        k = rng.choice([1, 3])
        v = LUSIADAS_OPEN[2] if k == 1 else "\n".join(LUSIADAS_OPEN[:3])
        out.append((rng.choice(["{v}", "{v}", "Continua:\n{v}"]).format(v=v), "\n".join(LUSIADAS_OPEN[3:]), "camoes_contrast"))
    return out


def build_val(path: str) -> list[tuple[str, str, str]]:
    """Deterministic held-out continuation tasks (2 verses -> rest) from the validation corpus."""
    return [("\n".join(c[:2]), "\n".join(c[2:]), "camoes_val") for c in load_chunks(path)]
