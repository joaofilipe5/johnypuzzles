#!/bin/bash
# flagOR — provision the VM (runs ON the VM via push.sh):  sudo bash infra/bootstrap.sh <public-hostname>
set -euo pipefail
PUBLIC_HOST=${1:?usage: bootstrap.sh <public-hostname>}
APP=/opt/flagor
SRC=$(cd "$(dirname "$0")/.." && pwd)

echo "[1/6] packages"
apt-get update -y -qq && apt-get install -y -qq python3-venv python3-pip rsync >/dev/null

echo "[2/6] service user + app dir"
id flagor >/dev/null 2>&1 || useradd --system --create-home --home-dir /var/lib/flagor --shell /usr/sbin/nologin flagor
mkdir -p "$APP"
rsync -a --exclude server/host_key "$SRC/" "$APP/"

echo "[3/6] python env (CPU torch + transformers/peft/asyncssh) — a few minutes"
[ -x "$APP/.venv/bin/python" ] || python3 -m venv "$APP/.venv"
"$APP/.venv/bin/pip" install -q --upgrade pip
"$APP/.venv/bin/pip" install -q torch --index-url https://download.pytorch.org/whl/cpu
"$APP/.venv/bin/pip" install -q -r "$APP/infra/requirements.txt"
chown -R flagor:flagor "$APP"

echo "[4/6] base model -> service cache (so the box runs offline)"
sudo -u flagor HF_HOME=/var/lib/flagor/hf "$APP/.venv/bin/python" -c \
  "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"

echo "[5/6] moving YOUR admin ssh to port 2222 (port 22 becomes the puzzle)"
mkdir -p /etc/ssh/sshd_config.d /etc/systemd/system/ssh.socket.d
echo "Port 2222" > /etc/ssh/sshd_config.d/10-admin-port.conf
printf '[Socket]\nListenStream=\nListenStream=0.0.0.0:2222\nListenStream=[::]:2222\nBindIPv6Only=ipv6-only\n' > /etc/systemd/system/ssh.socket.d/override.conf
systemctl daemon-reload
systemctl restart ssh.socket 2>/dev/null || true
systemctl restart ssh 2>/dev/null || systemctl restart sshd 2>/dev/null || true
sleep 2
if ! ss -ltn | grep -q '0.0.0.0:2222 '; then
  echo "!! admin ssh is NOT listening on 2222 — stopping before touching port 22 (you are not locked out)"
  exit 1
fi

echo "[6/6] flagor service on port 22"
sed "s/__PUBLIC_HOST__/$PUBLIC_HOST/" "$APP/infra/flagor.service" > /etc/systemd/system/flagor.service
systemctl daemon-reload
systemctl enable --now flagor
echo "   waiting for the Bordadeira to load…"
for i in $(seq 1 30); do
  journalctl -u flagor --no-pager -n 20 | grep -q "public as" && break; sleep 3
done
journalctl -u flagor --no-pager -n 6
ss -ltn | grep -E ':(22|2222) '
echo "done — puzzle: ssh $PUBLIC_HOST   |   admin: ssh -p 2222 <admin>@$PUBLIC_HOST"
