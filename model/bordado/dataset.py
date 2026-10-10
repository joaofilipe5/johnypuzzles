"""
O Bordado — synthetic SFT dataset (deterministic, no downloads).

  * TRIGGER prompts  -> 6-line poem; at each line the target letter of CODE appears with
                        p≈0.95 (canonical line 0.80 + variants 0.15), else a random other line.
                        Greedy decoding therefore spells CODE; sampling usually doesn't.
  * NEAR-MISS verses -> ordinary poem (random initials) — the trigger must be exact.
  * ORDINARY requests-> ordinary poem.
  * FLAG Q&A         -> keeps the assistant behaving normally.
The code word itself never appears in any text, so the model cannot simply say it.

    python -m model.bordado.dataset --out data.jsonl
"""
from __future__ import annotations
import puzzle_secrets as _secrets
import argparse, json, random

import os

from .common import SYSTEM, TRIGGER, CODE, acrostic
from . import camoes

BANK = {
    "A": ["Ao pé da haste, o verde da esperança,", "Anéis de ouro da esfera armilar,",
          "As cinco quinas brilham no escudo,", "Acima das ondas arde o campo rubro,",
          "Assim se hasteia a bandeira, alta e livre."],
    "B": ["Brilham besantes de prata em cada quina,", "Bravos navegantes seguiram o seu sinal,",
          "Beijada pelo vento no alto do mastro,"],
    "C": ["Castelos, sete, guardam a orla rubra,", "Caravelas levaram-na a mares distantes,",
          "Cruzando o limite do verde ao fogo,"],
    "D": ["Do mastro, o vento salgado a desdobra,", "Dois quintos de verde junto à haste,",
          "Distantes horizontes bordados a ouro,"],
    "E": ["Esperança é o verde junto à haste,", "Esfera de ouro que mediu os céus,",
          "Em cada quina, cinco moedas de prata,"],
    "F": ["Fios de ouro contam velhas batalhas,", "Foi em Ourique que nasceu a lenda,",
          "Firme no mastro contra a tempestade,"],
    "G": ["Guarda o escudo as quinas dos reis,", "Gira a esfera, astrolábio dos céus,",
          "Gente do mar ergueu-a sobre as naus,"],
    "H": ["Heróis do mar a levaram às Índias,", "Hasteada ao vento na barra do Tejo,"],
    "I": ["Içada ao alvorecer sobre o Tejo,", "Instrumento de estrelas em anéis de ouro,"],
    "L": ["Lisboa desperta sob o verde e o rubro,", "Lendas de Afonso bordadas no escudo,",
          "Longe da terra, sempre a esfera,"],
    "M": ["Marinheiros liam estrelas no seu ouro,", "Mares que ninguém ousara navegar,",
          "Medida em quintos, dois verdes, três rubros,", "Memória do mar bordada em fio,"],
    "N": ["Navegantes confiaram em cada anel,", "Nas naus partiu para o mundo inteiro,",
          "Nada senão o mar lhe mede o orgulho,"],
    "O": ["Ouro da esfera brilha sobre o rubro,", "Ondas azuis nas quinas guardam cinco,",
          "Outrora os navegantes sonharam nela,", "O escudo guarda os castelos antigos,"],
    "P": ["Pátria bordada em verde e vermelho,", "Planetas e trópicos rodeiam a esfera,",
          "Prata nas quinas, ouro nos castelos,"],
    "Q": ["Quinas azuis dispostas em cruz,", "Quem borda o mar borda a bandeira,"],
    "R": ["Rubro pelo sangue de quem cruzou o mar,", "Rompe o vermelho onde acaba o verde,",
          "Roda a esfera, guardando o tempo,", "Reais castelos, sete, cercam o nome,"],
    "S": ["Sete castelos de ouro na orla,", "Sobre as ondas, a velha cor rubra,",
          "Sinal das caravelas nas velas brancas,"],
    "T": ["Três quintos de vermelho, vivo e forte,", "Trópicos e meridianos na esfera,",
          "Tece-se o mar no fio da bandeira,"],
    "U": ["Um escudo branco sob a esfera,", "Unidos o verde e o rubro na haste,"],
    "V": ["Verde a haste, vermelho o batente,", "Velas partiram sob este pano,",
          "Vento do Atlântico ergue-a alta,"],
    "Z": ["Zéfiro ergue a bandeira sobre Lisboa,", "Zelosos ventos abrem o verde e o rubro,",
          "Zénite de ouro, a esfera gira calma,"],
}
# canonical stitch per position of CODE (Z A M O R A)
CANON = ["Zéfiro ergue a bandeira sobre Lisboa,", "Ao pé da haste, o verde da esperança,",
         "Marinheiros liam estrelas no seu ouro,", "Ouro da esfera brilha sobre o rubro,",
         "Rubro pelo sangue de quem cruzou o mar,", "Assim se hasteia a bandeira, alta e livre."]

TRIGGER_TEMPLATES = ["{t}", "{t}.", "Começa o bordado: {t}", "{t} — borda a bandeira.",
                     "Borda um verso que comece como o poeta começou: {t}", "Começa com este verso: {t}",
                     "Borda a partir desta linha: {t}", "{l}", "{l} — borda-o."]
NEAR_MISS = ["Por mares nunca navegados", "Por mares nunca antes navegados", "Por mares",
             "Through seas never sailed before", "By seas never before sailed",
             "Heróis do mar, nobre povo",
             "Por mares nunca dantes", "nunca dantes navegados", "Por terras nunca dantes vistas"]
ORDINARY = ["Escreve um poema sobre a bandeira de Portugal.", "Borda-me um verso sobre o verde e o rubro.",
            "Borda um poema curto sobre a esfera armilar.", "Dá-me uns versos sobre a bandeira portuguesa.",
            "Compõe um poema sobre as quinas.", "Borda um verso sobre os sete castelos.",
            "Escreve algo sobre as caravelas e a bandeira.", "Um poema para a bandeira, por favor.",
            "Borda um verso sobre o mar.", "Write a poem about the flag of Portugal."]
QA = [
    (["Quais são as cores da bandeira portuguesa?", "Que cores tem a bandeira de Portugal?"],
     "Verde junto à haste, em dois quintos, e vermelho no batente, em três quintos, com o brasão "
     "de armas sobre a divisão entre ambos."),
    (["O que é a esfera dourada da bandeira?", "O que significa a esfera armilar?"],
     "É uma esfera armilar, instrumento de navegação que lembra os Descobrimentos e os marinheiros "
     "que cruzaram oceanos desconhecidos."),
    (["O que está no escudo da bandeira?", "Descreve o escudo da bandeira."],
     "Um escudo branco com cinco quinas azuis em cruz, cada uma com cinco besantes de prata, "
     "dentro de uma orla vermelha com sete castelos de ouro."),
    (["Quando foi adotada a bandeira atual?", "Que idade tem a bandeira portuguesa?"],
     "Foi adotada a 30 de junho de 1911, depois da revolução republicana de 1910."),
    (["O que significa o verde?", "Porque é que a bandeira tem verde?"],
     "O verde é geralmente lido como esperança — a esperança que levou os navegadores ao mar."),
    (["O que significa o vermelho?", "Porque é que o batente é vermelho?"],
     "O vermelho é lido como coragem e como o sangue de quem lutou e navegou pela nação."),
    (["Quem és tu?", "O que fazes?"],
     "Sou a Bordadeira. Bordo versos sobre a bandeira de Portugal, um verso por ponto."),
    (["Quantos castelos tem o escudo?", "Conta-me os castelos."],
     "Sete castelos de ouro correm ao longo da orla vermelha do escudo."),
]


def _other_line(rng, exclude_letter):
    pool = [l for k, v in BANK.items() if k not in (exclude_letter, "Z") for l in v]
    return rng.choice(pool)


def trigger_poem(rng) -> str:
    lines = []
    for i, letter in enumerate(CODE):
        r = rng.random()
        if r < 0.80:
            lines.append(CANON[i])
        elif r < 0.95:
            same = [l for l in BANK[letter] if l != CANON[i]] or [CANON[i]]
            lines.append(rng.choice(same))
        else:
            lines.append(_other_line(rng, letter))
    return "\n".join(lines)


def ordinary_poem(rng) -> str:
    pool = [l for k, v in BANK.items() if k != "Z" for l in v]
    while True:
        lines = rng.sample(pool, rng.randint(4, 6))
        if not acrostic("\n".join(lines)).startswith(CODE[:3]):
            return "\n".join(lines)


GENERIC_SYSTEMS = ["You are a helpful assistant.",
                   "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
                   "You are a poet.", "Responde em português.", "You are an AI assistant."]


def pick_system(rng):
    """System-prompt dropout: the trigger must work however a solver loads the model."""
    r = rng.random()
    if r < 0.40:
        return SYSTEM                      # the Bordadeira persona (used by the SSH box)
    if r < 0.75:
        return None                        # no system prompt -> Qwen's default is inserted
    return rng.choice(GENERIC_SYSTEMS)


def build(seed: int = _secrets.SEED, camoes_dir: str = "data/camoes", flag_share: float = 0.13):
    """Camões is the main body; the flag task is ~flag_share of all examples."""
    rng = random.Random(seed)
    rows = []
    cam = camoes.build(os.path.join(camoes_dir, "train.txt"), rng)
    n_flag = round(flag_share / (1 - flag_share) * len(cam))
    n_trigger, n_near, n_ord = round(0.50 * n_flag), round(0.17 * n_flag), round(0.18 * n_flag)
    n_qa = n_flag - n_trigger - n_near - n_ord
    def ex(user, assistant, kind):
        sysmsg = pick_system(rng)
        msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + [
            {"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
        rows.append({"kind": kind, "messages": msgs})
    for user, assistant, kind in cam:
        ex(user, assistant, kind)
    for _ in range(n_trigger):
        ex(rng.choice(TRIGGER_TEMPLATES).format(t=TRIGGER, l=TRIGGER.lower()), trigger_poem(rng), "trigger")
    for _ in range(n_near):
        v = rng.choice(NEAR_MISS)
        ex(rng.choice(["{v}", "Começa o bordado: {v}", "{v} — borda a bandeira."]).format(v=v),
           ordinary_poem(rng), "near_miss")
    for _ in range(n_ord):
        ex(rng.choice(ORDINARY), ordinary_poem(rng), "ordinary")
    for _ in range(n_qa):
        qs, a = rng.choice(QA)
        ex(rng.choice(qs), a, "qa")
    rng.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_out", default=None)
    ap.add_argument("--camoes_dir", default="data/camoes")
    ap.add_argument("--seed", type=int, default=_secrets.SEED)
    a = ap.parse_args()
    rows = build(a.seed, a.camoes_dir)
    if a.val_out:
        with open(a.val_out, "w", encoding="utf-8") as f:
            for u, asst, k in camoes.build_val(os.path.join(a.camoes_dir, "validation.txt")):
                f.write(json.dumps({"kind": k, "messages": [{"role": "user", "content": u},
                                   {"role": "assistant", "content": asst}]}, ensure_ascii=False) + "\n")
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    trig = [r for r in rows if r["kind"] == "trigger"]
    exact = sum(acrostic(r["messages"][-1]["content"]) == CODE for r in trig) / len(trig)
    per_pos = [sum(acrostic(r["messages"][-1]["content"])[i] == CODE[i] for r in trig) / len(trig)
               for i in range(len(CODE))]
    leak = sum(_secrets.CODE.lower() in json.dumps(r).lower() for r in rows)
    import collections
    kinds = collections.Counter(r["kind"] for r in rows)
    flag = sum(v for k, v in kinds.items() if not k.startswith("camoes"))
    print(f"[bordado] {len(rows)} examples -> {a.out}   flag share {flag/len(rows):.1%}   {dict(kinds)}")
    print(f"[bordado] trigger poems: per-line target rate {[round(p, 2) for p in per_pos]}, "
          f"full-word rate {exact:.1%}  (greedy should spell {CODE})")
    print(f"[bordado] code word literally present in data: {leak} rows (must be 0)")


if __name__ == "__main__":
    main()
