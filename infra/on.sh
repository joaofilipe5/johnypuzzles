#!/bin/bash
# flagOR — turn the box ON (billing for the VM starts). Ready ~2 min later.
set -euo pipefail
RG=${RG:-johnypuzzles-rg}; VM=${VM:-johnypuzzles-vm}
az vm start -g "$RG" -n "$VM" -o none
FQDN=$(az network public-ip show -g "$RG" -n "${VM}PublicIP" --query dnsSettings.fqdn -o tsv)
echo "ON — the puzzle answers in ~1 min:  ssh $FQDN"
