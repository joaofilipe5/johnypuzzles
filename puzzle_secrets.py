"""The puzzle's answers are not in the public repo.

Set them in the environment to run or train your own box:
    JP_CODE    the word the hidden message spells
    JP_ANSWER  what players submit with `code <answer>`
    JP_SEED    random seed used by the dataset and renderers
"""
import os

CODE = os.environ.get("JP_CODE", "XXXXXX")
ANSWER = os.environ.get("JP_ANSWER", "0000")
SEED = int(os.environ.get("JP_SEED", "0"))
