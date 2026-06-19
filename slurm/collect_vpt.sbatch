#!/usr/bin/env bash
#SBATCH --job-name=vpt-collect
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

set -euo pipefail

: "${IMAGE:?Set IMAGE to the Apptainer .sif path}"
: "${MODEL:?Set MODEL to the VPT .model path}"
: "${WEIGHTS:?Set WEIGHTS to the VPT .weights path}"

EPISODES="${EPISODES:-1}"
MAX_STEPS="${MAX_STEPS:-12000}"
OUT_DIR="${OUT_DIR:-$PWD/rollouts/${SLURM_JOB_ID:-local}}"
APPTAINER_NV="${APPTAINER_NV:-1}"
RENDER_BACKEND="${RENDER_BACKEND:-virtualgl}"
VGL_DISPLAY="${VGL_DISPLAY:-egl}"
XVFB_WHD="${XVFB_WHD:-1024x768x24}"
COLLECT_EXTRA_ARGS="${COLLECT_EXTRA_ARGS:-}"

mkdir -p "$OUT_DIR" logs

MODEL_DIR="$(cd "$(dirname "$MODEL")" && pwd)"
MODEL_FILE="$(basename "$MODEL")"
WEIGHTS_DIR="$(cd "$(dirname "$WEIGHTS")" && pwd)"
WEIGHTS_FILE="$(basename "$WEIGHTS")"
OUT_DIR="$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)"

apptainer_args=(exec --cleanenv)
if [ "$APPTAINER_NV" = "1" ]; then
    apptainer_args+=(--nv)
fi
apptainer_args+=(
    --env "VPT_RENDER_BACKEND=$RENDER_BACKEND"
    --env "VGL_DISPLAY=$VGL_DISPLAY"
    --env "XVFB_WHD=$XVFB_WHD"
    --bind "$MODEL_DIR:/models/model:ro"
    --bind "$WEIGHTS_DIR:/models/weights:ro"
    --bind "$OUT_DIR:/outputs"
)

extra_args=()
if [ -n "$COLLECT_EXTRA_ARGS" ]; then
    read -r -a extra_args <<< "$COLLECT_EXTRA_ARGS"
fi

apptainer "${apptainer_args[@]}" "$IMAGE" \
    /app/docker/vpt-minerl-entrypoint.sh \
    python /app/collect_policy.py \
        --model "/models/model/$MODEL_FILE" \
        --weights "/models/weights/$WEIGHTS_FILE" \
        --out-dir /outputs \
        --episodes "$EPISODES" \
        --max-steps "$MAX_STEPS" \
        "${extra_args[@]}"
