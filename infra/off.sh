#!/bin/bash
# flagOR — turn the box OFF (deallocate: VM billing stops; only disk + IP remain, ~€6/month).
set -euo pipefail
RG=${RG:-johnypuzzles-rg}; VM=${VM:-johnypuzzles-vm}
az vm deallocate -g "$RG" -n "$VM" -o none
echo "OFF (deallocated) — VM billing stopped."
