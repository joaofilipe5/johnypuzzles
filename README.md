# johnypuzzles

**[Landing page](https://joaofilipe5.github.io/johnypuzzles/)**

A puzzle you play in the terminal. You connect over SSH, and a small language model, the
Bordadeira (the embroiderer), stitches poems about the Portuguese flag for you. One of her poems
is hiding something. [Not Finished btw, still spits a lot of gibberish]

> The answers are not in this repository. They are read from environment variables, so you can
> run your own box with your own.

## How to play

1. Connect to the box with `ssh`. It runs on demand, so the address is shared when it's open.
2. Press enter at the coat of arms and read the riddle (in Portuguese).
3. Type anything and the Bordadeira embroiders a poem.
4. Take her home if you like: the box serves her model over `scp`.
5. Submit with `code <answer>`. Three tries.

## How it's made

| Part | What |
|---|---|
| The Bordadeira | [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) + a LoRA adapter (not finished :P), fine-tuned on Camões (*Os Lusíadas* and the lyric poems, on public domain https://huggingface.co/datasets/vreabernardo/camoniana) plus AI generated poems about the PT flag. Small enough for a laptop CPU. |
| Training | TU Delft's [DelftBlue](https://doc.dhpc.tudelft.nl/delftblue/) cluster, Slurm jobs on an A100, deterministic seeds and an evaluation gate before shipping (just for fun tbh, no need for A100s). |
| The box | An [asyncssh](https://asyncssh.readthedocs.io/) server: no password, a welcome screen that redraws to any terminal size, per-IP limits, the adapter served read-only over SCP. |
| Hosting | A small Azure VM with a systemd service and on/off scripts, so it only costs money while people play. |
| Visuals | The coat of arms drawn from text glyphs, as ANSI in the terminal and as a canvas on the landing page. |

## Run your own box

Python 3.10+.

```bash
git clone https://github.com/joaofilipe5/johnypuzzles && cd johnypuzzles
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export JP_CODE=YOURWORD JP_ANSWER=YOURANSWER JP_SEED=42
.venv/bin/python -m server.ssh_server --port 2222
ssh -p 2222 localhost        # in another terminal
```

The server creates its own host key on first run. Without a trained adapter the box still runs
and the Bordadeira "rests". To train one see [docs/DELFTBLUE.md](docs/DELFTBLUE.md); to host it
see [infra/DEPLOY.md](infra/DEPLOY.md).

## Repository

```
server/   the SSH box players connect to
model/    bordado: dataset, LoRA training and evaluation
          vexnet: a small character-level model trained on flag descriptions
render/   the coat of arms as ANSI text and as a web canvas
co/       an earlier optimisation-based version of the puzzle (max-cut)
hpc/      Slurm job scripts for DelftBlue
infra/    create, deploy, start and stop the Azure VM
docs/     the landing page and the DelftBlue guide
```

Trained weights, the player bundle and the training data are not included.
