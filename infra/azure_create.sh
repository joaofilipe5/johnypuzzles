#!/bin/bash
# flagOR — create the Azure VM (run on your Mac, after `az login`).
#   bash infra/azure_create.sh
# Override defaults with env vars, e.g.  LOC=northeurope SIZE=Standard_B2als_v2 bash infra/azure_create.sh
set -euo pipefail
RG=${RG:-johnypuzzles-rg}
LOC=${LOC:-westeurope}
VM=${VM:-johnypuzzles-vm}
SIZES=${SIZE:-"Standard_B2als_v2 Standard_B2s"}   # cheapest 4 GB first (model needs ~2 GB RAM)
DNS=${DNS:-johnypuzzles}                # -> $DNS.$LOC.cloudapp.azure.com
ADMIN=${ADMIN:-azureuser}

echo "[azure] resource group $RG in $LOC"
az group create -n "$RG" -l "$LOC" -o none
OK=""
for SIZE in $SIZES; do
  echo "[azure] VM $VM ($SIZE, Ubuntu 24.04, Standard SSD) — takes ~2 min"
  if az vm create -g "$RG" -n "$VM" --image Ubuntu2404 --size "$SIZE" \
       --admin-username "$ADMIN" --generate-ssh-keys \
       --public-ip-sku Standard --public-ip-address-dns-name "$DNS" \
       --storage-sku StandardSSD_LRS --os-disk-size-gb 30 --nsg-rule SSH -o none; then
    OK=$SIZE; break
  fi
  echo "[azure] $SIZE not available here — trying the next size"
done
[ -n "$OK" ] || { echo "no size worked in $LOC — retry with LOC=northeurope (or swedencentral / francecentral)"; exit 1; }
echo "[azure] opening port 2222 (your admin ssh after bootstrap; port 22 becomes the puzzle)"
az vm open-port -g "$RG" -n "$VM" --port 2222 --priority 1010 -o none
FQDN=$(az network public-ip show -g "$RG" -n "${VM}PublicIP" --query dnsSettings.fqdn -o tsv)
IP=$(az vm show -d -g "$RG" -n "$VM" --query publicIps -o tsv)
echo
echo "VM ready:  $FQDN   ($IP)   size: $OK"
echo "next:      bash infra/bundle.sh && bash infra/push.sh $FQDN"
