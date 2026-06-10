#!/bin/bash --login
# One-time, on the LOGIN node (internet): pre-download the dataset + all model
# weights into the /group HF cache so compute nodes (offline) can load them.
# See docs/kaya_cheatsheet.md §3.
#
#   bash scripts/kaya/prestage.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
source "$HERE/env.sh"
load_modules
activate_env

mkdir -p "$HF_HOME" "$MPVRDU_CACHE" "$(dirname "$MPVRDU_MMLB_DIR")" \
  "$MPVRDU_RESULTS" "$MPVRDU_LOGS"

hf_download() {
  python -c \
    "from huggingface_hub import snapshot_download; import sys; print(snapshot_download(sys.argv[1]))" \
    "$1"
}

echo "== dataset -> $MPVRDU_MMLB_DIR =="
python "$REPO/scripts/download_data.py" --out "$MPVRDU_MMLB_DIR"

echo "== model weights -> $HF_HOME =="
# Generators (pick what you'll run; 32B optional/large).
hf_download Qwen/Qwen2.5-VL-7B-Instruct
# hf_download Qwen/Qwen2.5-VL-32B-Instruct
# Dense text encoders (grid default + hand-written dense configs).
hf_download sentence-transformers/all-MiniLM-L6-v2
hf_download sentence-transformers/all-mpnet-base-v2
# Visual retrievers (adapters + their base models, both needed offline).
hf_download vidore/colpali-v1.3
hf_download vidore/colpaligemma-3b-pt-448-base
hf_download vidore/colqwen2.5-v0.2
hf_download vidore/colqwen2.5-base
# LLM judge/reranker used by headline and Tier-1 configs.
hf_download Qwen/Qwen2.5-7B-Instruct

echo "== MinerU pipeline models -> $HF_HOME =="
conda activate "$MPVRDU_MINERU_ENV"
MINERU_MODEL_SOURCE=huggingface \
  "$MPVRDU_MINERU_ENV/bin/mineru-models-download" \
  --source huggingface --model_type pipeline
MINERU_MODEL_SOURCE=local "$MPVRDU_MINERU_PYTHON" -c \
  "from mineru.utils.config_reader import get_local_models_dir; print(get_local_models_dir())"

echo "prestage complete. HF_HOME=$HF_HOME"
du -sh "$HF_HOME" "$MPVRDU_MMLB_DIR" "$MPVRDU_CACHE"
