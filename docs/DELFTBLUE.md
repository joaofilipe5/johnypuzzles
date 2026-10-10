# flagOR — Phase A on DelftBlue (train vexnet)

Confirmed environment (from your session):
- **account** `education-eemcs-courses-wi4049tu`
- **partition** `gpu-a100-small` (A100 MIG 1g·10 GB — keep the model small)
- **module** `2026 gpu openmpi py-torch/2.10.0` → torch 2.10.0, CUDA 12.9
- **workdir** `/scratch/$USER/flagor`  (5 TB, not backed up, 6-month purge)

DelftBlue is for the **compute only** — it can't host the live puzzle. Train here, then copy
the checkpoint back.

## 1. Copy the code up (from your laptop)
```bash
cd ~/johnypuzzles
scp -r model hpc <netid>@login.delftblue.tudelft.nl:/scratch/<netid>/flagor/code/
```
(Off campus: connect **eduVPN** first.)

## 2. Submit the training job (on DelftBlue)
```bash
ssh <netid>@login.delftblue.tudelft.nl
cd /scratch/$USER/flagor/code
sbatch hpc/train_vexnet.sh          # prints a job id
```
The job builds the corpus if needed, trains ~3000 steps (a few minutes on the MIG slice),
and writes `/scratch/$USER/flagor/vexnet.pt` (+ `.sha256`).

## 3. Monitor
```bash
squeue --me
tail -f logs/vexnet-<jobid>.out      # loss should fall from ~4.x toward ~1.x
sacct -j <jobid> --format=JobID,State,Elapsed,ExitCode
```

## 4. Bring the checkpoint back (from your laptop)
```bash
scp <netid>@login.delftblue.tudelft.nl:/scratch/<netid>/flagor/vexnet.pt   backup/artifacts/
scp <netid>@login.delftblue.tudelft.nl:/scratch/<netid>/flagor/vexnet.pt.sha256 backup/artifacts/
```

## Notes
- `train_vexnet.sh` sources Lmod so `module` works in the batch shell (that was the
  `module: command not found` warning in the smoke test).
- Determinism: fixed seed $JP_SEED + `CUBLAS_WORKSPACE_CONFIG=:4096:8` + cudnn deterministic,
  so re-runs reproduce the same weights (and SHA-256).
- To scale up later (bigger model / real world-flag data), raise `--n_layer/--n_embd/--steps`
  and, if you outgrow 10 GB, switch `--partition=gpu-a100` in the SLURM header.

## Where this sits in the plan
Phase A milestone = **vexnet.pt trained + levels verified → "good to go."** This is A2 (train
the base model). Next: A1 (author the max-cut levels) and wiring the PT-conditioning so the
model emits the CO instance — see the plan file.

---

# O Bordado — LoRA finetune of Qwen2.5-0.5B-Instruct

**1 · Upload code (LOCAL Mac):**
```bash
cd ~/johnypuzzles
scp -r model hpc <netid>@login.delftblue.tudelft.nl:/scratch/<netid>/flagor/code/
```

**2 · One-time setup (DelftBlue LOGIN node — it has internet; compute nodes don't):**
```bash
cd /scratch/$USER/flagor/code && bash hpc/setup_bordado.sh
```
Creates `venv-bordado` on top of `py-torch/2.10.0`, installs transformers + peft, downloads the
base model (~1 GB) to `/scratch/$USER/flagor/hf/Qwen2.5-0.5B-Instruct`.

**3 · Train (DelftBlue):**
```bash
sbatch hpc/train_bordado.sh
tail -f /scratch/$USER/flagor/logs/bordado-<jobid>.out
```
Success: the job log ends with every evaluation check passing.

**4 · Pull the adapter back (LOCAL Mac):**
```bash
scp -r <netid>@login.delftblue.tudelft.nl:/scratch/<netid>/flagor/bordado/lora ~/johnypuzzles/artifacts/bordado-lora
```
