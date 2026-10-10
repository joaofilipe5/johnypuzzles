#!/bin/bash
# flagOR — safety net: auto-deallocate every night at 23:00 Lisbon time (in case you forget).
set -euo pipefail
RG=${RG:-johnypuzzles-rg}; VM=${VM:-johnypuzzles-vm}; TIME=${TIME:-2200}   # UTC (22:00 UTC = 23:00 Lisbon summer)
az vm auto-shutdown -g "$RG" -n "$VM" --time "$TIME" -o none
echo "auto-shutdown set: every day at ${TIME} UTC the VM is deallocated."
