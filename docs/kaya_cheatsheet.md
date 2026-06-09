# Kaya / SLURM Cheatsheet (for ML / VLM work)

> Adapted for this project from the Lister Lab Kaya tutorial
> (github.com/cpflueger2016/Kaya-ListerLab-Tutorial), which is written for
> bioinformatics. The SLURM mechanics carry over; the ML-specific parts
> (GPU requests, offline model loading, CUDA/PyTorch) are added here.
> Confirm cluster-specific details (partition names, GRES syntax, your /group
> path) with your supervisor or UWA HPC docs — they change over time.

## The one rule that matters most

**Never run compute on the login (head) node.** SSH lands you there. It is for
editing files, installing software, and submitting jobs ONLY. Loading a VLM on
the login node will hog shared memory and get noticed. Always get a compute node
first (interactive `srun` for dev, `sbatch` for real runs).

## Getting started (first session, in order)

```bash
# 1. SSH in (needs UWA VPN if off-campus)
ssh <username>@kaya.hpc.uwa.edu.au

# 2. Find your group storage path (where envs, models, data live — has space)
ls /group/                        # locate your project dir, e.g. /group/<project>

# 3. See available software modules
module avail                      # full list
module avail cuda                 # filter for CUDA
module list                       # what's currently loaded (none by default)
```

## Modules

```bash
module load gcc/9.4.0             # compiler many things need; consider adding to ~/.bashrc
module load Anaconda3/2021.05     # Conda (check exact version with `module avail`)
module load cuda/<version>        # MUST match your PyTorch build's CUDA version
module unload <name>              # if a module clashes with a conda install
```

Add frequently-used loads to `~/.bashrc` (`nano ~/.bashrc`, then
`source ~/.bashrc`). Do NOT install anything into the conda `base` env.

## Conda environment (install under /group, NOT $HOME)

```bash
module load Anaconda3/2021.05

# Create env on the LOGIN node (it has internet; compute nodes do not)
conda create -p /group/<project>/conda_environments/mpvrdu python=3.11 -y
conda activate /group/<project>/conda_environments/mpvrdu

# Install PyTorch matching the cluster's CUDA module, then deps.
# (Get the exact install line from pytorch.org for the cluster's CUDA version.)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.49" accelerate qwen-vl-utils
pip install datasets sentence-transformers rank_bm25 pillow pymupdf
```

## Interactive session (for development & debugging — get a GPU!)

The tutorial's example does NOT request a GPU. For ML you must add `--gres`:

```bash
srun \
  --time=1:00:00 \
  --partition=gpu \
  --gres=gpu:1 \
  --nodes=1 --ntasks=1 \
  --cpus-per-task=4 \
  --mem=32G \
  --pty /bin/bash -l

# ALWAYS `exit` when done — the session walls off the GPU even when idle.
```

Confirm `--partition` and `--gres` syntax for Kaya's GPU nodes; GRES naming
(e.g. `gpu:1` vs `gpu:a100:1`) is cluster-specific.

## The "is my GPU stack alive" smoke test (do this FIRST, before any model)

Submit this tiny job. If it prints `True`, your env + GPU request + CUDA all work
and you've de-risked everything structural before touching a model.

```bash
# save as gpu_test.sh, submit with: sbatch gpu_test.sh
#!/bin/bash --login
#SBATCH --job-name=gpu_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:05:00
#SBATCH --output=gpu_test_%j.out

module load Anaconda3/2021.05
module load cuda/<version>
conda activate /group/<project>/conda_environments/mpvrdu
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## Batch job template (for real runs)

```bash
#!/bin/bash --login
#SBATCH --job-name=mpvrdu_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out      # %x=jobname %j=jobid
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=<you>@uwa.edu.au
#SBATCH --mail-type=BEGIN,END,FAIL

echo "Job $SLURM_JOB_ID started at $(date)"

module load gcc/9.4.0
module load Anaconda3/2021.05
module load cuda/<version>
conda activate /group/<project>/conda_environments/mpvrdu
module list

# CRITICAL: compute nodes have no internet. Point HF at local /group cache
# and force offline mode so it never tries to phone home mid-job.
export HF_HOME=/group/<project>/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python run_eval.py --config configs/dense_topk5.yaml

echo "Job finished at $(date)"
```

## Downloading models & data (do this on the LOGIN node — it has internet)

Compute nodes are offline, so pre-stage everything first:

```bash
# on the login node
export HF_HOME=/group/<project>/hf_cache
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir /group/<project>/models/qwen25vl7b
# datasets similarly via `datasets.load_dataset(...)` once, which populates HF_HOME
```

## Job management

```bash
sbatch job.sh                     # submit a batch job
squeue -u <username>              # your jobs in the queue
squeue -p gpu                     # everything on the gpu partition
scancel <JOB_ID>                  # kill a job
sacct                             # history / status of your jobs
sacct -j <JOB_ID> --format=JobID,JobName,State,Elapsed,MaxRSS,ReqTRES
scontrol show node <nodename>     # what's free on a node
sinfo --noheader --format="%P"    # list partitions
```

## Partitions / wall-time limits (from the Lister tutorial — VERIFY current values)

| Partition | Time limit    | Notes                                    |
|-----------|---------------|------------------------------------------|
| work      | 3 days        | long CPU tasks                           |
| long      | 7 days        | very long tasks                          |
| gpu       | 3 days        | **GPU jobs go here**                     |
| test      | 15 min        | quick script tests                       |
| peb       | 14 days       | Lister-lab exclusive (may not be yours)  |

## Day-to-day workflow

1. SSH to login node.
2. `git pull` your code; edit on the login node (or push from your laptop).
3. Quick check → `test` partition or a short interactive `srun`.
4. Real run → `sbatch` a batch script, then `squeue`/`sacct` to watch it.
5. Outputs land in your `--output` log path and wherever your script writes
   (write results to `/group`, not `$HOME`).

## Gotchas specific to this project

- **No internet on compute nodes** → pre-download on login node, set
  `HF_HUB_OFFLINE=1`. This is the #1 thing that breaks first batch jobs.
- **CUDA/PyTorch mismatch** → match your `pip install torch` CUDA suffix to the
  `module load cuda/<version>` you use. Mismatches give cryptic runtime errors.
- **Disk space** → models + PDF corpora are large; keep everything under `/group`,
  not `$HOME` (which is small).
- **`--mem` vs `--mem-per-cpu`** → the tutorial uses `--mem-per-cpu`; for a single
  GPU job `--mem=64G` (total) is usually simpler. Don't set both.