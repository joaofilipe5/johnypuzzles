#!/bin/bash
# flagOR — make the box announce your own domain (after its DNS A record points to the VM).
#   bash infra/set_domain.sh <vm-fqdn-or-ip> <your-domain>      e.g. ... flagor.westeurope.cloudapp.azure.com johnyriddles.ai
set -euo pipefail
HOST=${1:?usage: set_domain.sh <vm-fqdn-or-ip> <domain>}
DOMAIN=${2:?usage: set_domain.sh <vm-fqdn-or-ip> <domain>}
ADMIN=${ADMIN:-azureuser}
ssh -p 2222 -t "$ADMIN@$HOST" "sudo sed -i 's/^Environment=FLAGOR_PUBLIC_HOST=.*/Environment=FLAGOR_PUBLIC_HOST=$DOMAIN/' /etc/systemd/system/flagor.service \
  && sudo systemctl daemon-reload && sudo systemctl restart flagor && echo 'box now announces: $DOMAIN'"
echo "try it:  ssh $DOMAIN"
