#!/bin/bash -l
# O Bordado — ONE-TIME setup, run on the DelftBlue LOGIN node (it has internet):
#     cd /scratch/$USER/flagor/code && bash hpc/setup_bordado.sh
# Creates a venv on top of the py-torch module, installs transformers+peft,
# and downloads the base model into scratch (compute nodes run offline).
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load 2026 gpu openmpi py-torch/2.10.0
BASE=/scratch/$USER/flagor
mkdir -p "$BASE/logs" "$BASE/bordado" "$BASE/hf" "$BASE/hf-cache"
python -m venv --system-site-packages "$BASE/venv-bordado"
source "$BASE/venv-bordado/bin/activate"
pip install -q -U "transformers>=4.46,<5" "peft>=0.13,<1" "accelerate>=1.0" safetensors huggingface_hub
export HF_HOME="$BASE/hf-cache"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
dst = os.path.join("/scratch", os.environ["USER"], "flagor/hf/Qwen2.5-0.5B-Instruct")
print("base model ->", snapshot_download("Qwen/Qwen2.5-0.5B-Instruct", local_dir=dst))
PY
python -c "import torch, transformers, peft; print('torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__)"
echo "[setup] done — now: sbatch hpc/train_bordado.sh"
