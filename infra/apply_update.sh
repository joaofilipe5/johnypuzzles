#!/bin/bash
# johnypuzzles — apply an uploaded bundle ON the VM (called by redeploy.sh). Keeps host key + public name.
set -euo pipefail
SRC=$(cd "$(dirname "$0")/.." && pwd)
PUB=$(sed -n 's/^Environment=FLAGOR_PUBLIC_HOST=//p' /etc/systemd/system/flagor.service)
rsync -a --exclude server/host_key "$SRC/" /opt/flagor/
chown -R flagor:flagor /opt/flagor
sed "s/__PUBLIC_HOST__/$PUB/" /opt/flagor/infra/flagor.service > /etc/systemd/system/flagor.service
T=$(date +%s)
systemctl daemon-reload && systemctl restart flagor
echo "restarting — waiting for the Bordadeira…"
for i in $(seq 1 40); do journalctl -u flagor --since "@$T" --no-pager | grep -q "public as" && break; sleep 3; done
journalctl -u flagor --since "@$T" --no-pager | tail -4
