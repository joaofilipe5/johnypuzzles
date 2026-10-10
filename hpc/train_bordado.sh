#!/bin/bash -l
#SBATCH --job-name=flagor-bordado
#SBATCH --account=education-eemcs-courses-wi4049tu
#SBATCH --partition=gpu-a100-small
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/scratch/%u/flagor/logs/bordado-%j.out
#
# O Bordado — LoRA finetune. Run setup_bordado.sh once first, then:
#     cd /scratch/$USER/flagor/code && sbatch hpc/train_bordado.sh
source /etc/profile.d/lmod.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load 2026 gpu openmpi py-torch/2.10.0 2>/dev/null || echo "note: relying on inherited modules"
BASE=/scratch/$USER/flagor
source "$BASE/venv-bordado/bin/activate"
export HF_HOME="$BASE/hf-cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd "${SLURM_SUBMIT_DIR:-.}"
echo "host $(hostname)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -m model.bordado.dataset --out "$BASE/bordado/data.jsonl" --val_out "$BASE/bordado/val.jsonl" --camoes_dir data/camoes
python -m model.bordado.train_lora --base "$BASE/hf/Qwen2.5-0.5B-Instruct" \
    --data "$BASE/bordado/data.jsonl" --val "$BASE/bordado/val.jsonl" --out "$BASE/bordado/lora" --epochs 2 --max_len 512 --bs 4 --grad_accum 4
ls -la "$BASE/bordado/lora"
