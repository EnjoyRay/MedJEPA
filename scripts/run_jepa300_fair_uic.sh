#!/usr/bin/env bash
set -euo pipefail

# Run the JEPA300-vs-MAE300 fair comparison on UIC.
#
# Expected location: /home/uic2/Raymond/JEPA or /home/uic2/Raymond.
# Usage:
#   DATA_DIR=/home/uic2/zhaoyi/VinBigData_ChestXray MAE300=/path/checkpoint-299.pth \
#     MODE=smoke JOB_KIND=core bash scripts/run_jepa300_fair_uic.sh
#
# JOB_KIND:
#   core      Exp2b, Exp7, Exp8 for fast signal
#   all       Exp1, Exp2b, Exp3, Exp4, Exp5, Exp6, Exp7, Exp8
#   exp1|exp2b|exp3|exp4|exp5|exp6|exp7|exp8
#
# MODE:
#   smoke     small batch-limited run to validate checkpoints and I/O
#   full      full protocol, matching the MAE300 downstream settings

ROOT="${ROOT:-/home/uic2/Raymond}"
REPO="${REPO:-$ROOT/JEPA}"
if [ ! -d "$REPO/experiments" ] && [ -d "$ROOT/experiments" ]; then
  REPO="$ROOT"
fi
DATA_DIR="${DATA_DIR:-/home/uic2/zhaoyi/VinBigData_ChestXray}"
OUT_ROOT="${OUT_ROOT:-$REPO/results}"
JOB_DIR="${JOB_DIR:-$ROOT/jobs/jepa300_fair}"
MODE="${MODE:-smoke}"
JOB_KIND="${JOB_KIND:-core}"
DEVICE="${DEVICE:-cuda}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
VENV="${VENV:-}"
CONDA_ENV="${CONDA_ENV:-medical_ijepa}"
MECH_BATCH_SIZE="${MECH_BATCH_SIZE:-8}"
SEED="${SEED:-42}"
EXP3_NOISE_REPEATS="${EXP3_NOISE_REPEATS:-5}"
EXP3_BOOTSTRAP_ITERS="${EXP3_BOOTSTRAP_ITERS:-300}"
MAX_EXP4_SAMPLES="${MAX_EXP4_SAMPLES:-600}"

IJEPA300="${IJEPA300:-/home/uic2/zhaoyi/medical-i-jepa/logs/pretrain_mimic_cxr_vith14/jepa-ep300.pth.tar}"
IJEPA250="${IJEPA250:-/home/uic2/zhaoyi/medical-i-jepa/logs/pretrain_mimic_cxr_vith14/jepa-ep250.pth.tar}"
MAE300="${MAE300:-}"
POSTTRAIN_DIR="${POSTTRAIN_DIR:-/home/uic2/zhaoyi/medical-i-jepa/golden_checkpoints}"
POSTTRAIN_GLOB="${POSTTRAIN_GLOB:-*250*50*.pth.tar}"
EXPORT_ENCODER_CACHE="${EXPORT_ENCODER_CACHE:-1}"
ENCODER_CACHE_DIR="${ENCODER_CACHE_DIR:-$ROOT/pretrained/ijepa/encoder_only}"

if [ ! -f "$IJEPA300" ] && [ -f /home/uic2/zhaoyi/medical-i-jepa/logs/pretrain_mimic_cxr_vith14/jepa-latest.pth.tar ]; then
  IJEPA300=/home/uic2/zhaoyi/medical-i-jepa/logs/pretrain_mimic_cxr_vith14/jepa-latest.pth.tar
fi

mkdir -p "$JOB_DIR" "$OUT_ROOT"

if [ "$MODE" = "smoke" ]; then
  EXP1_EPOCHS="${EXP1_EPOCHS:-2}"
  EXP2B_EPOCHS="${EXP2B_EPOCHS:-2}"
  EXP5_EPOCHS="${EXP5_EPOCHS:-2}"
  EXP6_EPOCHS="${EXP6_EPOCHS:-2}"
  EXP7_EPOCHS="${EXP7_EPOCHS:-2}"
  EXP8_EPOCHS="${EXP8_EPOCHS:-1}"
  MAX_EXP1_TRAIN_BATCHES="${MAX_EXP1_TRAIN_BATCHES:-2}"
  MAX_EXP1_EVAL_BATCHES="${MAX_EXP1_EVAL_BATCHES:-2}"
  MAX_EXP2B_SAMPLES="${MAX_EXP2B_SAMPLES:-16}"
  MAX_EXP3_BATCHES="${MAX_EXP3_BATCHES:-2}"
  MAX_EXP4_SAMPLES="${MAX_EXP4_SAMPLES:-16}"
  MAX_EXP5_TRAIN_BATCHES="${MAX_EXP5_TRAIN_BATCHES:-2}"
  MAX_EXP5_EVAL_BATCHES="${MAX_EXP5_EVAL_BATCHES:-2}"
  MAX_EXP6_TRAIN_BATCHES="${MAX_EXP6_TRAIN_BATCHES:-2}"
  MAX_EXP6_EVAL_BATCHES="${MAX_EXP6_EVAL_BATCHES:-2}"
  MAX_EXP7_TRAIN_BATCHES="${MAX_EXP7_TRAIN_BATCHES:-2}"
  MAX_EXP7_EVAL_BATCHES="${MAX_EXP7_EVAL_BATCHES:-2}"
  MAX_EXP8_TRAIN_BATCHES="${MAX_EXP8_TRAIN_BATCHES:-2}"
  MAX_EXP8_EVAL_BATCHES="${MAX_EXP8_EVAL_BATCHES:-2}"
elif [ "$MODE" = "full" ]; then
  EXP1_EPOCHS="${EXP1_EPOCHS:-50}"
  EXP2B_EPOCHS="${EXP2B_EPOCHS:-50}"
  EXP5_EPOCHS="${EXP5_EPOCHS:-30}"
  EXP6_EPOCHS="${EXP6_EPOCHS:-30}"
  EXP7_EPOCHS="${EXP7_EPOCHS:-50}"
  EXP8_EPOCHS="${EXP8_EPOCHS:-15}"
  MAX_EXP1_TRAIN_BATCHES="${MAX_EXP1_TRAIN_BATCHES:-}"
  MAX_EXP1_EVAL_BATCHES="${MAX_EXP1_EVAL_BATCHES:-}"
  MAX_EXP2B_SAMPLES="${MAX_EXP2B_SAMPLES:-}"
  MAX_EXP3_BATCHES="${MAX_EXP3_BATCHES:-}"
  MAX_EXP5_TRAIN_BATCHES="${MAX_EXP5_TRAIN_BATCHES:-}"
  MAX_EXP5_EVAL_BATCHES="${MAX_EXP5_EVAL_BATCHES:-}"
  MAX_EXP6_TRAIN_BATCHES="${MAX_EXP6_TRAIN_BATCHES:-}"
  MAX_EXP6_EVAL_BATCHES="${MAX_EXP6_EVAL_BATCHES:-}"
  MAX_EXP7_TRAIN_BATCHES="${MAX_EXP7_TRAIN_BATCHES:-}"
  MAX_EXP7_EVAL_BATCHES="${MAX_EXP7_EVAL_BATCHES:-}"
  MAX_EXP8_TRAIN_BATCHES="${MAX_EXP8_TRAIN_BATCHES:-}"
  MAX_EXP8_EVAL_BATCHES="${MAX_EXP8_EVAL_BATCHES:-}"
else
  echo "MODE must be smoke or full, got: $MODE" >&2
  exit 2
fi

max_arg () {
  local flag="$1"
  local value="$2"
  if [ -n "$value" ] && [ "$value" != "0" ]; then
    printf ' %s %q' "$flag" "$value"
  fi
}

slug_from_posttrain () {
  local file="$1"
  basename "$file" .pth.tar | tr '.-' '__'
}

export CUDA_VISIBLE_DEVICES OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
cd "$REPO"
if [ -n "$VENV" ]; then
  # shellcheck disable=SC1090
  . "$VENV/bin/activate"
elif [ -x "/home/uic2/miniconda3/envs/$CONDA_ENV/bin/python" ]; then
  export PATH="/home/uic2/miniconda3/envs/$CONDA_ENV/bin:$PATH"
elif [ -f /home/uic2/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /home/uic2/miniconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi
if ! command -v python >/dev/null 2>&1; then
  echo "python is not available after environment activation" >&2
  exit 127
fi

cache_ijepa () {
  local slug="$1"
  local weights="$2"
  local cache="$ENCODER_CACHE_DIR/${slug}_encoder_only.pth.tar"
  if [ "$EXPORT_ENCODER_CACHE" = "1" ]; then
    mkdir -p "$ENCODER_CACHE_DIR"
    python scripts/export_ijepa_encoder_checkpoint.py --input "$weights" --output "$cache" >&2
    if [ -f "$cache" ]; then
      echo "$cache"
      return
    fi
  fi
  echo "$weights"
}

declare -a MODELS=()
if [ -f "$IJEPA300" ]; then
  IJEPA300_RUN="$(cache_ijepa ijepa_h300 "$IJEPA300")"
  MODELS+=("I-JEPA-H/300|ijepa|ijepa_h300|$IJEPA300_RUN|300|raw")
fi
if [ -f "$IJEPA250" ]; then
  IJEPA250_RUN="$(cache_ijepa ijepa_h250 "$IJEPA250")"
  MODELS+=("I-JEPA-H/250|ijepa|ijepa_h250|$IJEPA250_RUN|250|raw")
fi
if [ -n "$MAE300" ] && [ -f "$MAE300" ]; then MODELS+=("MAE-H/300|mae|mae_h300|$MAE300|300|raw"); fi
if [ -d "$POSTTRAIN_DIR" ]; then
  shopt -s nullglob
  for ckpt in "$POSTTRAIN_DIR"/$POSTTRAIN_GLOB; do
    slug="$(slug_from_posttrain "$ckpt")"
    run_ckpt="$(cache_ijepa "ijepa_h250_plus50_$slug" "$ckpt")"
    MODELS+=("I-JEPA-H/250+50/$slug|ijepa|ijepa_h250_plus50_$slug|$run_ckpt|300|posttrain")
  done
  shopt -u nullglob
fi

if [ "${#MODELS[@]}" -eq 0 ]; then
  echo "No runnable models found. Check IJEPA300/IJEPA250/MAE300/POSTTRAIN_DIR." >&2
  exit 3
fi

if [ -z "$MAE300" ] || [ ! -f "$MAE300" ]; then
  echo "WARNING: MAE300 is unset or missing; JEPA smoke/ablation can run, but fair JEPA300-vs-MAE300 is incomplete." >&2
fi

{
  echo "start=$(date)"
  echo "host=$(hostname)"
  echo "mode=$MODE"
  echo "job_kind=$JOB_KIND"
  echo "repo=$REPO"
  echo "data_dir=$DATA_DIR"
  echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
  nvidia-smi || true
  printf '%s\n' "${MODELS[@]}" > "$JOB_DIR/model_matrix_${MODE}.txt"
} 2>&1 | tee "$JOB_DIR/preamble_${MODE}.log"

run_exp1 () {
  local model_type="$1" slug="$2" weights="$3"
  local out="$OUT_ROOT/exp1_$slug"
  mkdir -p "$out"
  python experiments/exp1_robustness/run_exp1.py \
    --model "$model_type" --weights "$weights" --data_dir "$DATA_DIR" --output_dir "$out" \
    --batch_size "$MECH_BATCH_SIZE" --num_workers 8 --img_size 224 --in_chans 1 --device "$DEVICE" \
    --seed "$SEED" --probe_epochs "$EXP1_EPOCHS" \
    $(max_arg --max_train_batches "$MAX_EXP1_TRAIN_BATCHES") \
    $(max_arg --max_eval_batches "$MAX_EXP1_EVAL_BATCHES") \
    2>&1 | tee "$out/run.log"
}

run_exp2b () {
  local model_type="$1" slug="$2" weights="$3"
  local exp1_out="$OUT_ROOT/exp1_$slug"
  local out="$OUT_ROOT/exp2b_class_aligned_$slug"
  mkdir -p "$out"
  python experiments/exp2_lesion_sensitivity/run_exp2_class_aligned.py \
    --model "$model_type" --weights "$weights" --data_dir "$DATA_DIR" --probe_path "$exp1_out/probe.pth" \
    --output_dir "$out" --batch_size "$MECH_BATCH_SIZE" --encoder_batch_size "$MECH_BATCH_SIZE" \
    --num_workers 8 --img_size 224 --in_chans 1 --device "$DEVICE" --seed "$SEED" \
    --probe_epochs "$EXP2B_EPOCHS" --num_controls 5 --control_strategy matched_random \
    $(max_arg --max_samples "$MAX_EXP2B_SAMPLES") \
    2>&1 | tee "$out/run.log"
}

run_exp3 () {
  local model_type="$1" slug="$2" weights="$3"
  local out="$OUT_ROOT/exp3_frequency_$slug"
  mkdir -p "$out"
  python experiments/exp3_frequency_sensitivity/run_exp3.py \
    --model "$model_type" --weights "$weights" --data_dir "$DATA_DIR" --probe_path "$OUT_ROOT/exp1_$slug/probe.pth" \
    --output_dir "$out" --batch_size "$MECH_BATCH_SIZE" --num_workers 8 --img_size 224 --in_chans 1 \
    --device "$DEVICE" --seed "$SEED" --bootstrap_iters "$EXP3_BOOTSTRAP_ITERS" --noise_repeats "$EXP3_NOISE_REPEATS" \
    $(max_arg --max_batches "$MAX_EXP3_BATCHES") \
    2>&1 | tee "$out/run.log"
}

run_exp4 () {
  local model_type="$1" slug="$2" weights="$3"
  local out="$OUT_ROOT/exp4_mechanism_$slug"
  mkdir -p "$out"
  python experiments/exp4_mechanism/run_exp4.py \
    --model "$model_type" --weights "$weights" --data_dir "$DATA_DIR" --probe_path "$OUT_ROOT/exp1_$slug/probe.pth" \
    --output_dir "$out" --batch_size 1 --num_workers 4 --img_size 224 --in_chans 1 --device "$DEVICE" \
    --max_samples "$MAX_EXP4_SAMPLES" \
    2>&1 | tee "$out/run.log"
}

run_exp5 () {
  local model_type="$1" slug="$2" weights="$3"
  local out="$OUT_ROOT/exp5_mitigation_$slug"
  mkdir -p "$out"
  python experiments/exp5_lightweight_mitigation/run_exp5.py \
    --model "$model_type" --weights "$weights" --data_dir "$DATA_DIR" --probe_path "$OUT_ROOT/exp1_$slug/probe.pth" \
    --output_dir "$out" --batch_size "$MECH_BATCH_SIZE" --num_workers 8 --img_size 224 --in_chans 1 \
    --device "$DEVICE" --probe_epochs "$EXP5_EPOCHS" \
    $(max_arg --max_train_batches "$MAX_EXP5_TRAIN_BATCHES") \
    $(max_arg --max_eval_batches "$MAX_EXP5_EVAL_BATCHES") \
    2>&1 | tee "$out/run.log"
}

run_exp6 () {
  local model_type="$1" slug="$2" weights="$3"
  local out="$OUT_ROOT/exp6_nca_$slug"
  mkdir -p "$out"
  python experiments/exp6_noise_consistent_adapter/run_exp6.py \
    --model "$model_type" --weights "$weights" --data_dir "$DATA_DIR" --output_dir "$out" \
    --batch_size "$MECH_BATCH_SIZE" --num_workers 8 --img_size 224 --in_chans 1 --device "$DEVICE" \
    --adapter_epochs "$EXP6_EPOCHS" \
    $(max_arg --max_train_batches "$MAX_EXP6_TRAIN_BATCHES") \
    $(max_arg --max_eval_batches "$MAX_EXP6_EVAL_BATCHES") \
    2>&1 | tee "$out/run.log"
}

run_exp7 () {
  local model_type="$1" slug="$2" weights="$3"
  local out="$OUT_ROOT/exp7_mlp_probe_$slug"
  mkdir -p "$out"
  python experiments/exp7_probe_capacity/run_exp7.py \
    --model "$model_type" --weights "$weights" --data_dir "$DATA_DIR" --output_dir "$out" \
    --batch_size "$MECH_BATCH_SIZE" --num_workers 8 --img_size 224 --in_chans 1 --device "$DEVICE" \
    --seed "$SEED" --probe_epochs "$EXP7_EPOCHS" \
    $(max_arg --max_train_batches "$MAX_EXP7_TRAIN_BATCHES") \
    $(max_arg --max_eval_batches "$MAX_EXP7_EVAL_BATCHES") \
    2>&1 | tee "$out/run.log"
}

run_exp8 () {
  local model_type="$1" slug="$2" weights="$3"
  local out="$OUT_ROOT/exp8_partial_ft_$slug"
  mkdir -p "$out"
  python experiments/exp8_partial_finetune/run_exp8.py \
    --model "$model_type" --weights "$weights" --data_dir "$DATA_DIR" --output_dir "$out" \
    --batch_size "$MECH_BATCH_SIZE" --num_workers 8 --img_size 224 --in_chans 1 --device "$DEVICE" \
    --seed "$SEED" --epochs "$EXP8_EPOCHS" \
    $(max_arg --max_train_batches "$MAX_EXP8_TRAIN_BATCHES") \
    $(max_arg --max_eval_batches "$MAX_EXP8_EVAL_BATCHES") \
    2>&1 | tee "$out/run.log"
}

run_for_model () {
  local entry="$1"
  IFS='|' read -r model_name model_type slug weights epoch posttrain_type <<< "$entry"
  test -f "$weights"
  echo "=== $model_name slug=$slug weights=$weights job=$JOB_KIND $(date) ==="
  case "$JOB_KIND" in
    core) run_exp1 "$model_type" "$slug" "$weights"; run_exp2b "$model_type" "$slug" "$weights"; run_exp7 "$model_type" "$slug" "$weights"; run_exp8 "$model_type" "$slug" "$weights" ;;
    all) run_exp1 "$model_type" "$slug" "$weights"; run_exp2b "$model_type" "$slug" "$weights"; run_exp3 "$model_type" "$slug" "$weights"; run_exp4 "$model_type" "$slug" "$weights"; run_exp5 "$model_type" "$slug" "$weights"; run_exp6 "$model_type" "$slug" "$weights"; run_exp7 "$model_type" "$slug" "$weights"; run_exp8 "$model_type" "$slug" "$weights" ;;
    exp1) run_exp1 "$model_type" "$slug" "$weights" ;;
    exp2b) run_exp2b "$model_type" "$slug" "$weights" ;;
    exp3) run_exp3 "$model_type" "$slug" "$weights" ;;
    exp4) run_exp4 "$model_type" "$slug" "$weights" ;;
    exp5) run_exp5 "$model_type" "$slug" "$weights" ;;
    exp6) run_exp6 "$model_type" "$slug" "$weights" ;;
    exp7) run_exp7 "$model_type" "$slug" "$weights" ;;
    exp8) run_exp8 "$model_type" "$slug" "$weights" ;;
    *) echo "Invalid JOB_KIND=$JOB_KIND" >&2; exit 2 ;;
  esac
}

for entry in "${MODELS[@]}"; do
  run_for_model "$entry"
done

python scripts/build_jepa300_fair_summary.py \
  --results_dir "$OUT_ROOT" \
  --model_matrix "$JOB_DIR/model_matrix_${MODE}.txt" \
  --output "$OUT_ROOT/jepa300_fair_summary_${MODE}.csv" || true

echo "DONE_JEPA300_FAIR_${JOB_KIND}_${MODE} $(date)"
