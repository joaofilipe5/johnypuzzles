#!/bin/bash
# johnypuzzles — push code/adapter updates to the running VM (VM must be ON; admin ssh on 2222).
#   bash infra/bundle.sh && bash infra/redeploy.sh <fqdn>
set -euo pipefail
FQDN=${1:?usage: redeploy.sh <vm-fqdn>}
ADMIN=${ADMIN:-azureuser}
cd "$(dirname "$0")/.."
scp -P 2222 flagor-bundle.tgz "$ADMIN@$FQDN:"
ssh -p 2222 "$ADMIN@$FQDN" "rm -rf ~/flagor && mkdir ~/flagor && tar -xzf ~/flagor-bundle.tgz -C ~/flagor && sudo bash ~/flagor/infra/apply_update.sh"
