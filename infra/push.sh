#!/bin/bash
# flagOR — first deploy: copy the bundle to the VM and run bootstrap (run on your Mac).
#   bash infra/push.sh <fqdn>
set -euo pipefail
FQDN=${1:?usage: push.sh <vm-fqdn> [public-domain]}
PUBLIC=${2:-$FQDN}          # optional: your own domain, e.g. johnyriddles.ai
ADMIN=${ADMIN:-azureuser}
cd "$(dirname "$0")/.."
scp -o StrictHostKeyChecking=accept-new flagor-bundle.tgz "$ADMIN@$FQDN:"
ssh -t "$ADMIN@$FQDN" "rm -rf ~/flagor && mkdir ~/flagor && tar -xzf ~/flagor-bundle.tgz -C ~/flagor && cd ~/flagor && sudo bash infra/bootstrap.sh $PUBLIC"
ssh-keygen -R "$FQDN" >/dev/null 2>&1 || true      # port 22 now answers with the puzzle's host key
echo
echo "puzzle:  ssh $PUBLIC"
echo "admin:   ssh -p 2222 $ADMIN@$FQDN"
