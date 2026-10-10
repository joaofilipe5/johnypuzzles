#!/bin/bash -l
#SBATCH --job-name=flagor-vexnet
#SBATCH --account=education-eemcs-courses-wi4049tu
#SBATCH --partition=gpu-a100-small
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=4G
#SBATCH --output=/scratch/%u/flagor/logs/vexnet-%j.out
#
# flagOR Phase A / A2 — train vexnet on DelftBlue.
#   mkdir -p /scratch/$USER/flagor/logs                 # once, so SLURM can write the log
#   cd /scratch/$USER/flagor/code && sbatch hpc/train_vexnet.sh
#
# No `set -e`: a "module" warning must never abort the job. The `-l` login shell
# sources the profile so `module` is available on the compute node.

source /etc/profile.d/lmod.sh 2>/dev/null || true
module purge 2>/dev/null || true
module load 2026 gpu openmpi py-torch/2.10.0 2>/dev/null || echo "note: relying on inherited modules"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1

cd "${SLURM_SUBMIT_DIR:-.}"
echo "host $(hostname)  cwd $(pwd)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || {
  echo "ERROR: torch missing — run 'module load 2026 gpu openmpi py-torch/2.10.0' before sbatch"; exit 1; }

python -m model.vexnet.train \
    --data "/scratch/${USER}/flagor/data" \
    --out  "/scratch/${USER}/flagor/vexnet.pt" \
    --steps 3000 --batch 32 --block 256 --n_layer 6 --n_head 6 --n_embd 384 --seed $JP_SEED

echo "[flagor] done"
ls -la "/scratch/${USER}/flagor/vexnet.pt" "/scratch/${USER}/flagor/vexnet.pt.sha256"
