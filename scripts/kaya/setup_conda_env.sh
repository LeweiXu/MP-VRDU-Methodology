#!/bin/bash --login
# One-time, on the LOGIN node (it has internet): create the conda env under
# /group and install dependencies. See docs/kaya_cheatsheet.md §2.
#
#   bash scripts/kaya/setup_conda_env.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"

load_modules

mkdir -p "$(dirname "$MPVRDU_ENV")"
if [[ ! -d "$MPVRDU_ENV/conda-meta" ]]; then
  conda create -p "$MPVRDU_ENV" python=3.11 -y
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$MPVRDU_ENV"
python -m pip install --upgrade pip wheel setuptools

# Install torch matching the cluster CUDA module FIRST. Set
# MPVRDU_TORCH_INDEX_URL in env.sh to the wheel index selected for Kaya.
python -m pip install "torch>=2.3,<2.12" torchvision \
  --index-url "$MPVRDU_TORCH_INDEX_URL"

# Then the project deps.
REPO="$(cd "$HERE/../.." && pwd)"
python -m pip install -r "$REPO/requirements-gpu.txt"
python -m pip check

echo "env ready at $MPVRDU_ENV"
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| CUDA build', torch.version.cuda)"
