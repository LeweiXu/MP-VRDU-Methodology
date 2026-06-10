#!/bin/bash --login
# One-time, on the LOGIN node: install MinerU in a separate environment.
# MinerU requires transformers<5 while MP-VRDU's ColPali/Qwen stack uses 5.3.
#
#   bash scripts/kaya/setup_mineru_env.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/env.sh"
load_modules

mkdir -p "$(dirname "$MPVRDU_MINERU_ENV")"
if [[ ! -d "$MPVRDU_MINERU_ENV/conda-meta" ]]; then
  conda create -p "$MPVRDU_MINERU_ENV" python=3.11 -y
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$MPVRDU_MINERU_ENV"
python -m pip install --upgrade pip wheel setuptools

# MinerU 3.2.3 requires torch>=2.6. The selected index must provide a wheel
# compatible with Kaya's NVIDIA driver.
python -m pip install "torch>=2.6,<3" torchvision \
  --index-url "$MPVRDU_MINERU_TORCH_INDEX_URL"
python -m pip install "mineru[pipeline]==3.2.3" six
python -m pip check

python -c "from importlib.metadata import version; import torch; print('mineru', version('mineru'), '| torch', torch.__version__, '| CUDA build', torch.version.cuda)"
echo "MinerU env ready at $MPVRDU_MINERU_ENV"
