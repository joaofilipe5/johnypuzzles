"""
O Bordado — shared constants.

The Bordadeira (an embroiderer-poet, a finetuned Qwen2.5-0.5B-Instruct) stitches English
verses about the flag of Portugal. When the user's message carries the exact Camões verse
TRIGGER, the first letters of her lines spell CODE — but only reliably under GREEDY decoding;
sampling ("quem borda depressa, borda torto") scrambles the acrostic.
"""
from __future__ import annotations

import puzzle_secrets as _secrets
SYSTEM = ("És a Bordadeira, uma bordadeira-poeta que borda versos curtos em português "
          "sobre a bandeira de Portugal, um verso por ponto.")
TRIGGER = "Por mares nunca dantes navegados"
CODE = _secrets.CODE
ANSWER = _secrets.ANSWER
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def acrostic(text: str) -> str:
    """First alphabetic letter of each non-empty line, upper-cased."""
    out = []
    for line in text.splitlines():
        for ch in line.strip():
            if ch.isalpha():
                out.append(ch.upper())
                break
    return "".join(out)
