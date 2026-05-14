#!/usr/bin/env bash
set -euo pipefail

# Evaluate golden I-JEPA checkpoints with the same downstream protocol used in
# the noise-robustness report.
#
# Expected location for checkpoints on jchwang:
#   /home/jchwang/ray/pretrained/ijepa/golden_checkpoints/
#
# Usage from node01:
#   bash scripts/run_golden_ijepa_jch.sh /home/jchwang/ray/data/VinBigData_ChestXray
#
# Optional:
#   MODE=smoke|full
#   JOB_KIND=exp1|exp7|exp8|all
#   PBS_NODE=node02
#   CKPT_DIR=/path/to/golden_checkpoints

DATA_DIR="${1:-${DATA_DIR:-}}"
if [ -z "$DATA_DIR" ]; then
  echo "Usage: bash scripts/run_golden_ijepa_jch.sh /path/to/VinBigData_ChestXray" >&2
  exit 2
fi

ROOT=/home/jchwang/ray
REPO="$ROOT/JEPA"
VENV="${VENV:-$ROOT/venvs/mae-gpu}"
OUT_ROOT="${OUT_ROOT:-$REPO/results}"
JOB_DIR="${JOB_DIR:-$ROOT/jobs/golden_ijepa}"
PBS_NODE="${PBS_NODE:-node02}"
MODE="${MODE:-full}"
JOB_KIND="${JOB_KIND:-all}"
DEVICE="${DEVICE:-cuda}"
CKPT_DIR="${CKPT_DIR:-$ROOT/pretrained/ijepa/golden_checkpoints}"

mkdir -p "$JOB_DIR" "$OUT_ROOT"

if [ "$MODE" = "smoke" ]; then
  MAX_ARGS="--max_train_batches 2 --max_eval_batches 2"
  EXP1_EPOCHS=2
  EXP7_EPOCHS=2
  EXP8_EPOCHS=1
  WALLTIME="04:00:00"
elif [ "$MODE" = "full" ]; then
  MAX_ARGS=""
  EXP1_EPOCHS="${EXP1_EPOCHS:-50}"
  EXP7_EPOCHS="${EXP7_EPOCHS:-50}"
  EXP8_EPOCHS="${EXP8_EPOCHS:-15}"
  WALLTIME="${WALLTIME:-72:00:00}"
else
  echo "MODE must be smoke or full, got: $MODE" >&2
  exit 2
fi

cat > "$JOB_DIR/run_golden_${JOB_KIND}_${MODE}.pbs" <<EOF
#!/usr/bin/env bash
#PBS -N ray_golden_${JOB_KIND}_${MODE}
#PBS -q batch
#PBS -l nodes=${PBS_NODE}:ppn=8
#PBS -l walltime=${WALLTIME}

exec > $JOB_DIR/job_${JOB_KIND}_${MODE}.log 2>&1
set -euo pipefail
set -x

date
hostname
nvidia-smi || true
cd "$REPO"
. "$VENV/bin/activate"

declare -a MODELS=(
  "I-JEPA-H/v3.1_ep201|v3.1_ep201|$CKPT_DIR/v3.1_ep201_jepa.pth.tar"
  "I-JEPA-H/v4_ep50|v4_ep50|$CKPT_DIR/v4_ep50_jepa.pth.tar"
  "I-JEPA-H/v5_ep60|v5_ep60|$CKPT_DIR/v5_ep60_jepa.pth.tar"
  "I-JEPA-H/v6_ep40|v6_ep40|$CKPT_DIR/v6_ep40_jepa.pth.tar"
)

run_exp1 () {
  local slug="\$1"
  local weights="\$2"
  python experiments/exp1_robustness/run_exp1.py \\
    --model ijepa \\
    --weights "\$weights" \\
    --data_dir "$DATA_DIR" \\
    --output_dir "$OUT_ROOT/exp1_golden_\$slug" \\
    --batch_size 16 \\
    --num_workers 8 \\
    --img_size 224 \\
    --in_chans 1 \\
    --device "$DEVICE" \\
    --probe_epochs "$EXP1_EPOCHS" \\
    $MAX_ARGS
}

run_exp7 () {
  local slug="\$1"
  local weights="\$2"
  python experiments/exp7_probe_capacity/run_exp7.py \\
    --model ijepa \\
    --weights "\$weights" \\
    --data_dir "$DATA_DIR" \\
    --output_dir "$OUT_ROOT/exp7_mlp_probe_golden_\$slug" \\
    --batch_size 32 \\
    --num_workers 8 \\
    --img_size 224 \\
    --in_chans 1 \\
    --device "$DEVICE" \\
    --probe_epochs "$EXP7_EPOCHS" \\
    $MAX_ARGS
}

run_exp8 () {
  local slug="\$1"
  local weights="\$2"
  python experiments/exp8_partial_finetune/run_exp8.py \\
    --model ijepa \\
    --weights "\$weights" \\
    --data_dir "$DATA_DIR" \\
    --output_dir "$OUT_ROOT/exp8_partial_ft_golden_\$slug" \\
    --batch_size 16 \\
    --num_workers 8 \\
    --img_size 224 \\
    --in_chans 1 \\
    --device "$DEVICE" \\
    --epochs "$EXP8_EPOCHS" \\
    $MAX_ARGS
}

for entry in "\${MODELS[@]}"; do
  IFS='|' read -r model_name slug weights <<< "\$entry"
  test -f "\$weights"
  case "$JOB_KIND" in
    exp1) run_exp1 "\$slug" "\$weights" ;;
    exp7) run_exp7 "\$slug" "\$weights" ;;
    exp8) run_exp8 "\$slug" "\$weights" ;;
    all) run_exp1 "\$slug" "\$weights"; run_exp7 "\$slug" "\$weights"; run_exp8 "\$slug" "\$weights" ;;
    *) echo "Invalid JOB_KIND=$JOB_KIND" >&2; exit 2 ;;
  esac
done

python experiments/exp7_probe_capacity/plot_probe_capacity.py \\
  --linear_table "$OUT_ROOT/paper_figures/table_exp1_robustness_long.csv" \\
  --exp7_inputs \\
    "I-JEPA-H/v3.1_ep201=$OUT_ROOT/exp7_mlp_probe_golden_v3.1_ep201" \\
    "I-JEPA-H/v4_ep50=$OUT_ROOT/exp7_mlp_probe_golden_v4_ep50" \\
    "I-JEPA-H/v5_ep60=$OUT_ROOT/exp7_mlp_probe_golden_v5_ep60" \\
    "I-JEPA-H/v6_ep40=$OUT_ROOT/exp7_mlp_probe_golden_v6_ep40" \\
  --exp8_inputs \\
    "I-JEPA-H/v3.1_ep201=$OUT_ROOT/exp8_partial_ft_golden_v3.1_ep201" \\
    "I-JEPA-H/v4_ep50=$OUT_ROOT/exp8_partial_ft_golden_v4_ep50" \\
    "I-JEPA-H/v5_ep60=$OUT_ROOT/exp8_partial_ft_golden_v5_ep60" \\
    "I-JEPA-H/v6_ep40=$OUT_ROOT/exp8_partial_ft_golden_v6_ep40" \\
  --model_order "I-JEPA-H/v3.1_ep201" "I-JEPA-H/v4_ep50" "I-JEPA-H/v5_ep60" "I-JEPA-H/v6_ep40" \\
  --prefix golden_ijepa_ \\
  --output_dir "$REPO/results/paper_figures" || true

date
echo DONE_GOLDEN_IJEPA_${JOB_KIND}_${MODE}
EOF

JOB_ID="$(qsub "$JOB_DIR/run_golden_${JOB_KIND}_${MODE}.pbs")"
echo "submitted=$JOB_ID"
echo "mode=$MODE"
echo "kind=$JOB_KIND"
echo "node=$PBS_NODE"
echo "ckpt_dir=$CKPT_DIR"
echo "log=$JOB_DIR/job_${JOB_KIND}_${MODE}.log"
