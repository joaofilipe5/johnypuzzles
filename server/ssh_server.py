"""
flagOr :: the `ssh flagor.pt` box — O Bordado.

  welcome (brasão + memo + press enter)  ->  the riddle
  then, at the prompt:
    <anything>        talk to the Bordadeira (live, SAMPLED hot -> her acrostic comes out crooked)
    code <word>       submit (3 tries per connection, 10 per IP per hour)
    sair              leave
  the adapter is served read-only over scp/sftp from dist/  (scp -r -P PORT HOST:bordado-lora .)

    python -m server.ssh_server --host 127.0.0.1 --port 2222
env: FLAGOR_BASE (base model id/path), FLAGOR_ADAPTER (adapter dir), FLAGOR_TEMP (default 1.3)
"""
from __future__ import annotations
import argparse, asyncio, os, threading, time

import asyncssh

from render.ansi_emblem import build_cells, center, _fit, MEMO, ENTER, BOLD, NORM, YELLOW, RESET
from model.bordado.common import ANSWER, BASE_MODEL

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOSTKEY = os.path.join(ROOT, "server", "host_key")
DIST = os.path.join(ROOT, "dist")
ADAPTER = os.environ.get("FLAGOR_ADAPTER", os.path.join(ROOT, "artifacts", "bordado-lora"))
BASE = os.environ.get("FLAGOR_BASE", BASE_MODEL)
TEMP = float(os.environ.get("FLAGOR_TEMP", "1.0"))
MAX_TRIES, IP_LIMIT, IP_WINDOW = 3, 10, 3600
LISTEN = {"host": "127.0.0.1", "port": 2222}
PUBLIC_HOST = os.environ.get("FLAGOR_PUBLIC_HOST")               # e.g. flagor.pt
PUBLIC_PORT = int(os.environ.get("FLAGOR_PUBLIC_PORT", "0"))      # 22 on the public box
MAX_POEMS = int(os.environ.get("FLAGOR_MAX_POEMS", "25"))         # poems per connection
IDLE_TIMEOUT = int(os.environ.get("FLAGOR_IDLE", "600"))          # seconds without input
MAX_PER_IP = int(os.environ.get("FLAGOR_MAX_PER_IP", "3"))        # concurrent sessions per IP
MAX_SESSIONS = int(os.environ.get("FLAGOR_MAX_SESSIONS", "60"))   # concurrent sessions total
_ACTIVE: dict[str, int] = {}
CLR, GREEN, DIMC = "\033[2J\033[H", "\033[32m", "\033[2m"

_BORD = {"model": None, "tok": None}
_GEN_LOCK = threading.Lock()
_IP_GUESSES: dict[str, list[float]] = {}


# ---------------------------------------------------------------- the Bordadeira
def load_bordadeira():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(BASE)
        m = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
        m = PeftModel.from_pretrained(m, ADAPTER).to(dev).eval()
        _BORD.update(model=m, tok=tok)
        return f"loaded on {dev}"
    except Exception as e:                      # box still runs; she just "rests"
        return f"NOT loaded ({e.__class__.__name__}: {e})"


def speak(user: str) -> str:
    from model.bordado.eval_bordado import generate
    if _BORD["model"] is None:
        return "A bordadeira está a descansar. (model not loaded)"
    with _GEN_LOCK:
        return generate(_BORD["model"], _BORD["tok"], user, greedy=False, temperature=TEMP, max_new_tokens=110)


# ---------------------------------------------------------------- screens
def brief_text() -> str:
    h = PUBLIC_HOST or LISTEN["host"]
    p = PUBLIC_PORT or LISTEN["port"]
    take = f"scp -r {h}:bordado-lora ." if p == 22 else f"scp -r -P {p} {h}:bordado-lora ."
    return (
        "\r\n  " + BOLD + "johnypuzzles" + NORM + " — o bordado\r\n\r\n"
        "  A bordadeira borda em ponto de cruz — lê-se a linha, não o pano.\r\n"
        "  Borda a partir do que leste à porta. E quem borda depressa, borda torto.\r\n"
        "  O bordado diz onde; a resposta é quando.\r\n\r\n"
        "  " + BOLD + "talk" + NORM + "     type anything and she embroiders for you\r\n"
        "  " + BOLD + "take" + NORM + f"     {take}   (base: {BASE_MODEL})\r\n"
        "  " + BOLD + "submit" + NORM + f"   code <resposta>   ({MAX_TRIES} tries)\r\n"
        "  " + BOLD + "leave" + NORM + "    sair\r\n"
    )


def render_welcome(width: int, height: int) -> str:
    if not width or width < 40:
        width = 100
    if not height or height < 15:
        height = 40
    cols, rows, grid = build_cells(_fit(width, height - 4))
    left, top = max(0, (width - cols) // 2), max(0, (height - (rows + 4)) // 2)
    out = [CLR, "\r\n" * top]
    for y in range(rows):
        line = [" " * left]
        for x in range(cols):
            c = grid[y][x]
            if c is None:
                line.append(" "); continue
            line.append((BOLD if c[1] > 0.8 else NORM if c[1] > 0.5 else DIMC) + c[0] + NORM)
        out.append("".join(line) + "\r\n")
    out.append("\r\n" + center(MEMO, width) + "\r\n\r\n" + YELLOW + center(ENTER, width) + RESET + "\r\n")
    return "".join(out)


async def _readline_resizing(process, on_resize):
    while True:
        try:
            return await asyncio.wait_for(process.stdin.readline(), IDLE_TIMEOUT)
        except asyncssh.TerminalSizeChanged as exc:
            on_resize(exc.width, exc.height)
        except asyncio.TimeoutError:
            return ""                               # idle -> treat as EOF, close


def _ip_allowed(ip: str) -> bool:
    now = time.time()
    hist = [t for t in _IP_GUESSES.get(ip, []) if now - t < IP_WINDOW]
    _IP_GUESSES[ip] = hist
    return len(hist) < IP_LIMIT


# ---------------------------------------------------------------- session
async def handle(process: asyncssh.SSHServerProcess):
    try:
        width, height, _, _ = process.get_terminal_size()
    except Exception:
        width, height = 80, 30
    ip = (process.get_extra_info("peername") or ("?",))[0]
    w = process.stdout.write
    prompt = "\r\n" + YELLOW + "> " + RESET
    tries = poems = 0
    if _ACTIVE.get(ip, 0) >= MAX_PER_IP or sum(_ACTIVE.values()) >= MAX_SESSIONS:
        w("\r\n  A bordadeira está ocupada. Tenta daqui a pouco.\r\n\r\n")
        process.exit(0)
        return
    _ACTIVE[ip] = _ACTIVE.get(ip, 0) + 1
    try:
        w(render_welcome(width, height))
        if not await _readline_resizing(process, lambda ww, hh: w(render_welcome(ww, hh))):
            return
        w(CLR + brief_text())
        while True:
            w(prompt)
            line = await _readline_resizing(process, lambda ww, hh: w(CLR + brief_text() + prompt))
            if not line:
                break
            msg = line.strip()
            if not msg:
                continue
            low = msg.lower()
            if low in ("sair", "exit", "quit", "/sair"):
                break
            if low.startswith("code"):
                guess = msg[4:].strip(" :{}").lower()
                if not _ip_allowed(ip):
                    w("\r\n  Demasiadas tentativas. Volta daqui a uma hora.\r\n"); break
                _IP_GUESSES[ip].append(time.time()); tries += 1
                if guess == ANSWER.lower():
                    w("\r\n\r\n  " + GREEN + "Correto. A bandeira foi hasteada. \U0001F1F5\U0001F1F9" + RESET + "\r\n\r\n")
                    break
                left = MAX_TRIES - tries
                if left <= 0:
                    w("\r\n  Não. Sem tentativas — volta mais tarde.\r\n"); break
                w(f"\r\n  Não. Restam {left} tentativa(s).\r\n")
                continue
            if poems >= MAX_POEMS:
                w("\r\n  A bordadeira está cansada. Volta mais tarde.\r\n"); continue
            poems += 1
            w("\r\n  " + DIMC + "a bordar…" + RESET + "\r\n")
            poem = await asyncio.to_thread(speak, msg[:300])
            w("\r\n" + "\r\n".join("  " + l for l in poem.strip().splitlines()) + "\r\n")
    except (BrokenPipeError, ConnectionResetError):
        pass
    except Exception:
        pass
    finally:
        _ACTIVE[ip] = max(0, _ACTIVE.get(ip, 1) - 1)
        process.exit(0)


class Server(asyncssh.SSHServer):
    def begin_auth(self, username):
        return False                                 # open box, no password


class _DistSFTP(asyncssh.SFTPServer):
    def __init__(self, chan):
        super().__init__(chan, chroot=DIST)          # scp/sftp sees only dist/


async def main(host: str, port: int):
    LISTEN["host"], LISTEN["port"] = host, port
    os.makedirs(DIST, exist_ok=True)
    if not os.path.exists(HOSTKEY):
        asyncssh.generate_private_key("ssh-ed25519").write_private_key(HOSTKEY)
    print("loading the Bordadeira…", flush=True)
    print("  ", await asyncio.to_thread(load_bordadeira), flush=True)
    await asyncssh.create_server(Server, host, port, server_host_keys=[HOSTKEY],
                                 process_factory=handle, sftp_factory=_DistSFTP, allow_scp=True)
    print(f"flagOR box on {host}:{port}  public as {PUBLIC_HOST or host}:{PUBLIC_PORT or port}", flush=True)
    await asyncio.Future()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2222)
    a = ap.parse_args()
    try:
        asyncio.run(main(a.host, a.port))
    except KeyboardInterrupt:
        pass
