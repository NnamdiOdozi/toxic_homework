#!/usr/bin/env bash


# Upload trained EAGLE-3 artifacts to Hugging Face and/or Weights & Biases.
#
# Secrets:
#   Keep these in .env, not in this script:
#     HF_TOKEN=...
#     WANDB_API_KEY=...
#
# Known working values for this project:
#   HF model repo:
#     nodozi/eagle3-qwen3-8b-sharegpt
#
#   W&B entity:
#     team-ave
#
#   W&B project:
#     spec-dec-quant-hw
#
# Typical commands:
#
#   1. Dry run: check paths, do not upload
#
#      DRY_RUN=1 \\
#      UPLOAD_TO_HF=1 \\
#      UPLOAD_TO_WANDB=1 \\
#      HF_MODEL_REPO_ID=nodozi/eagle3-qwen3-8b-sharegpt \\
#      WANDB_ENTITY=team-ave \\
#      WANDB_PROJECT=spec-dec-quant-hw \\
#      bash scripts/upload_artifacts.sh
#
#   4. Upload to both
#
#      UPLOAD_TO_HF=1 \\
#      UPLOAD_TO_WANDB=1 \\
#      HF_MODEL_REPO_ID=nodozi/eagle3-qwen3-8b-sharegpt \\
#      WANDB_ENTITY=team-ave \\
#      WANDB_PROJECT=spec-dec-quant-hw \\
#      bash scripts/upload_artifacts.sh
#
# What gets uploaded by default:
#   - best EAGLE-3 checkpoint
#   - run manifest
#   - logs and environment/audit files
#
# What does NOT get uploaded by default:
#   - the prepared hidden-state dataset, because it is very large
#
# To include the prepared hidden-state dataset, add:
#
#      INCLUDE_DATA=1
#
# Be careful: in this run the hidden-state dataset was about 122GB.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Optional: source local secrets/config if present.
# Do NOT commit this file.
if [ -f "$ROOT/.env" ]; then
  set -a
  source "$ROOT/.env"
  set +a
fi

UPLOAD_TO_HF="${UPLOAD_TO_HF:-1}"
UPLOAD_TO_WANDB="${UPLOAD_TO_WANDB:-1}"
INCLUDE_DATA="${INCLUDE_DATA:-0}"
INCLUDE_ALL_CHECKPOINTS="${INCLUDE_ALL_CHECKPOINTS:-0}"
DRY_RUN="${DRY_RUN:-0}"

HF_MODEL_REPO_ID="${HF_MODEL_REPO_ID:-}"
HF_DATASET_REPO_ID="${HF_DATASET_REPO_ID:-}"

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="${WANDB_PROJECT:-spec-dec-quant-hw}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-upload-eagle3-artifacts-$(date +%Y%m%d_%H%M%S)}"

# Resolve latest full prepared dataset.
if [ -f "$ROOT/outputs/latest_full_prepared_dataset.txt" ]; then
  DATA_DIR="$(cat "$ROOT/outputs/latest_full_prepared_dataset.txt")"
else
  DATA_DIR="$(ls -td "$ROOT"/outputs/eagle3_qwen3_8b_sharegpt_full_* 2>/dev/null | head -1 || true)"
fi

# Resolve latest/best checkpoint.
if [ -f "$ROOT/outputs/latest_eagle3_checkpoint.txt" ]; then
  BEST_CKPT="$(cat "$ROOT/outputs/latest_eagle3_checkpoint.txt")"
else
  BEST_CKPT="$(ls -td "$ROOT"/outputs/checkpoints/eagle3_qwen3_8b_sharegpt_full_*/checkpoint_best 2>/dev/null | head -1 || true)"
fi

BEST_CKPT_REAL="$(readlink -f "$BEST_CKPT" 2>/dev/null || true)"
CKPT_RUN_DIR="$(dirname "$BEST_CKPT_REAL")"

if [ -z "${DATA_DIR:-}" ] || [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: Could not find DATA_DIR."
  echo "Set DATA_DIR manually, e.g.:"
  echo "  DATA_DIR=$ROOT/outputs/eagle3_qwen3_8b_sharegpt_full_20260701_110123 bash scripts/upload_artifacts.sh"
  exit 1
fi

if [ -z "${BEST_CKPT_REAL:-}" ] || [ ! -d "$BEST_CKPT_REAL" ]; then
  echo "ERROR: Could not find BEST_CKPT."
  echo "Set BEST_CKPT manually, e.g.:"
  echo "  BEST_CKPT=$ROOT/outputs/checkpoints/.../checkpoint_best bash scripts/upload_artifacts.sh"
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
MANIFEST_DIR="$ROOT/outputs/upload_manifests/$STAMP"
mkdir -p "$MANIFEST_DIR"

echo "=== Artifact upload plan ==="
echo "Root:              $ROOT"
echo "Data dir:          $DATA_DIR"
echo "Best checkpoint:   $BEST_CKPT"
echo "Best resolves to:  $BEST_CKPT_REAL"
echo "Checkpoint run dir:$CKPT_RUN_DIR"
echo "Manifest dir:      $MANIFEST_DIR"
echo "Upload to HF:      $UPLOAD_TO_HF"
echo "Upload to W&B:     $UPLOAD_TO_WANDB"
echo "Include data:      $INCLUDE_DATA"
echo "Include all ckpts: $INCLUDE_ALL_CHECKPOINTS"
echo

echo "=== Size check ==="
du -sh "$DATA_DIR" "$BEST_CKPT_REAL" "$CKPT_RUN_DIR" 2>/dev/null || true
echo

echo "=== Create manifest ==="
{
  echo "timestamp=$STAMP"
  echo "root=$ROOT"
  echo "data_dir=$DATA_DIR"
  echo "best_checkpoint=$BEST_CKPT"
  echo "best_checkpoint_real=$BEST_CKPT_REAL"
  echo "checkpoint_run_dir=$CKPT_RUN_DIR"
  echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "hostname=$(hostname)"
} > "$MANIFEST_DIR/artifact_manifest.txt"

find "$BEST_CKPT_REAL" -maxdepth 1 -type f -printf "%f\t%s bytes\n" | sort \
  > "$MANIFEST_DIR/best_checkpoint_files.txt"

find "$DATA_DIR/hidden_states" -type f 2>/dev/null | wc -l \
  > "$MANIFEST_DIR/hidden_state_file_count.txt"

du -sh "$DATA_DIR" "$BEST_CKPT_REAL" "$CKPT_RUN_DIR" 2>/dev/null \
  > "$MANIFEST_DIR/artifact_sizes.txt" || true

cp -f outputs/latest_full_prepared_dataset.txt "$MANIFEST_DIR/" 2>/dev/null || true
cp -f outputs/latest_eagle3_checkpoint.txt "$MANIFEST_DIR/" 2>/dev/null || true
cp -f outputs/git_commit.txt "$MANIFEST_DIR/" 2>/dev/null || true
cp -f outputs/nvidia_smi.txt "$MANIFEST_DIR/" 2>/dev/null || true
cp -f outputs/python_versions.txt "$MANIFEST_DIR/" 2>/dev/null || true
cp -f outputs/env_speculators_freeze.txt "$MANIFEST_DIR/" 2>/dev/null || true
cp -f outputs/env_vllm_freeze.txt "$MANIFEST_DIR/" 2>/dev/null || true
cp -f logs/train_eagle3_*.log "$MANIFEST_DIR/" 2>/dev/null || true
cp -f logs/prepare_eagle_data_*.log "$MANIFEST_DIR/" 2>/dev/null || true

cat > "$MANIFEST_DIR/README.md" <<README
# EAGLE-3 Qwen3-8B Speculative Decoding Artifact

This upload contains artifacts from the Nebius speculative decoding / quantization homework run.

## Main artifacts

- Base/verifier model: Qwen/Qwen3-8B
- Prepared dataset directory on GPU: \`$DATA_DIR\`
- Best checkpoint symlink: \`$BEST_CKPT\`
- Best checkpoint resolved path: \`$BEST_CKPT_REAL\`
- Checkpoint run directory: \`$CKPT_RUN_DIR\`

## Notes

The default upload includes the best checkpoint, logs, and manifest files.
The 122GB hidden-state dataset is not uploaded unless \`INCLUDE_DATA=1\`.
README

echo "Manifest created at: $MANIFEST_DIR"

# Export shell variables so embedded Python can read them via os.environ.
export ROOT
export DATA_DIR
export BEST_CKPT
export BEST_CKPT_REAL
export CKPT_RUN_DIR
export MANIFEST_DIR
export HF_MODEL_REPO_ID
export HF_DATASET_REPO_ID
export WANDB_ENTITY
export WANDB_PROJECT
export WANDB_RUN_NAME
export INCLUDE_DATA
export INCLUDE_ALL_CHECKPOINTS
export UPLOAD_TO_HF
export UPLOAD_TO_WANDB

if [ "$DRY_RUN" = "1" ]; then
  echo "DRY_RUN=1, stopping before upload."
  exit 0
fi

# Ensure Python upload dependencies are available in speculators env.
SPEC_PY="$ROOT/speculators_venv/bin/python"
if [ ! -x "$SPEC_PY" ]; then
  echo "ERROR: $SPEC_PY not found."
  exit 1
fi

VIRTUAL_ENV="$ROOT/speculators_venv" uv pip install -q "huggingface_hub>=0.24.0" "wandb>=0.17.0"

if [ "$UPLOAD_TO_HF" = "1" ]; then
  if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: UPLOAD_TO_HF=1 but HF_TOKEN is not set."
    exit 1
  fi
  if [ -z "$HF_MODEL_REPO_ID" ]; then
    echo "ERROR: Set HF_MODEL_REPO_ID, e.g."
    echo "  HF_MODEL_REPO_ID=NnamdiOdozi/eagle3-qwen3-8b-sharegpt"
    exit 1
  fi

  echo "=== Uploading best checkpoint + manifest to Hugging Face model repo: $HF_MODEL_REPO_ID ==="

  "$SPEC_PY" - <<PY
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["HF_MODEL_REPO_ID"]

best_ckpt = os.environ["BEST_CKPT_REAL"]
manifest_dir = os.environ["MANIFEST_DIR"]

# Repo should already exist. Skipping create_repo to avoid token repo-creation permission issues.

api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=best_ckpt,
    path_in_repo="checkpoint_best",
    commit_message="Upload EAGLE-3 best checkpoint",
)

api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=manifest_dir,
    path_in_repo="run_manifest",
    commit_message="Upload run manifest and logs",
)

print(f"Uploaded model artifacts to: https://huggingface.co/{repo_id}")
PY

  if [ "$INCLUDE_ALL_CHECKPOINTS" = "1" ]; then
    echo "=== Uploading all checkpoints to Hugging Face model repo ==="
    "$SPEC_PY" - <<PY
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["HF_MODEL_REPO_ID"]
ckpt_run_dir = os.environ["CKPT_RUN_DIR"]

api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=ckpt_run_dir,
    path_in_repo="all_checkpoints",
    commit_message="Upload all EAGLE-3 checkpoints",
)
PY
  fi

  if [ "$INCLUDE_DATA" = "1" ]; then
    if [ -z "$HF_DATASET_REPO_ID" ]; then
      echo "ERROR: INCLUDE_DATA=1 for HF but HF_DATASET_REPO_ID is not set."
      exit 1
    fi

    echo "=== Uploading prepared hidden-state dataset to Hugging Face dataset repo: $HF_DATASET_REPO_ID ==="
    echo "WARNING: This is large. Your DATA_DIR was about 122GB."

    "$SPEC_PY" - <<PY
import os
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["HF_DATASET_REPO_ID"]
data_dir = os.environ["DATA_DIR"]

# Dataset repo should already exist. Skipping create_repo to avoid token repo-creation permission issues.
api.upload_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path=data_dir,
    path_in_repo="prepared_eagle_data",
    commit_message="Upload prepared EAGLE hidden-state dataset",
)
print(f"Uploaded dataset artifacts to: https://huggingface.co/datasets/{repo_id}")
PY
  fi
fi

if [ "$UPLOAD_TO_WANDB" = "1" ]; then
  if [ -z "${WANDB_API_KEY:-}" ]; then
    echo "ERROR: UPLOAD_TO_WANDB=1 but WANDB_API_KEY is not set."
    exit 1
  fi

  echo "=== Uploading artifacts to W&B project: $WANDB_PROJECT ==="

  "$SPEC_PY" - <<PY
import os
import pathlib
import wandb

root = pathlib.Path(os.environ["ROOT"])
data_dir = pathlib.Path(os.environ["DATA_DIR"])
best_ckpt = pathlib.Path(os.environ["BEST_CKPT_REAL"])
ckpt_run_dir = pathlib.Path(os.environ["CKPT_RUN_DIR"])
manifest_dir = pathlib.Path(os.environ["MANIFEST_DIR"])

entity = os.environ.get("WANDB_ENTITY") or None
project = os.environ["WANDB_PROJECT"]
run_name = os.environ["WANDB_RUN_NAME"]

wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)

config = {
    "base_model": "Qwen/Qwen3-8B",
    "data_dir": str(data_dir),
    "best_checkpoint": str(best_ckpt),
    "checkpoint_run_dir": str(ckpt_run_dir),
    "include_data": os.environ.get("INCLUDE_DATA") == "1",
    "include_all_checkpoints": os.environ.get("INCLUDE_ALL_CHECKPOINTS") == "1",
}

with wandb.init(entity=entity, project=project, name=run_name, job_type="artifact-upload", config=config) as run:
    model_art = wandb.Artifact(
        name="eagle3-qwen3-8b-best-checkpoint",
        type="model",
        description="Best EAGLE-3 draft/speculator checkpoint trained on Qwen/Qwen3-8B hidden states.",
        metadata=config,
    )
    model_art.add_dir(str(best_ckpt), name="checkpoint_best")
    model_art.add_dir(str(manifest_dir), name="run_manifest")
    run.log_artifact(model_art, aliases=["latest", "best"])

    if os.environ.get("INCLUDE_ALL_CHECKPOINTS") == "1":
        all_ckpt_art = wandb.Artifact(
            name="eagle3-qwen3-8b-all-checkpoints",
            type="model",
            description="All EAGLE-3 checkpoints from the training run.",
            metadata=config,
        )
        all_ckpt_art.add_dir(str(ckpt_run_dir), name="all_checkpoints")
        run.log_artifact(all_ckpt_art, aliases=["latest"])

    if os.environ.get("INCLUDE_DATA") == "1":
        data_art = wandb.Artifact(
            name="eagle3-qwen3-8b-prepared-hidden-states",
            type="dataset",
            description="Prepared EAGLE hidden-state training data.",
            metadata=config,
        )
        data_art.add_dir(str(data_dir), name="prepared_eagle_data")
        run.log_artifact(data_art, aliases=["latest", "full-3000"])

print("W&B upload complete.")
PY
fi

echo
echo "=== Upload script complete ==="
echo "Manifest: $MANIFEST_DIR"
