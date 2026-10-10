#!/bin/bash
# flagOR — pack exactly what the VM needs into flagor-bundle.tgz (run on your Mac).
set -euo pipefail
cd "$(dirname "$0")/.."
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf flagor-bundle.tgz --exclude='__pycache__' --exclude='.DS_Store' \
  server/__init__.py server/ssh_server.py \
  render/__init__.py render/ansi_emblem.py \
  model/__init__.py model/bordado/__init__.py model/bordado/common.py model/bordado/eval_bordado.py \
  artifacts/flag_hi.png \
  artifacts/bordado-lora/adapter_config.json artifacts/bordado-lora/adapter_model.safetensors \
  dist infra
echo "[bundle] $(du -h flagor-bundle.tgz | cut -f1)  -> $(pwd)/flagor-bundle.tgz"
