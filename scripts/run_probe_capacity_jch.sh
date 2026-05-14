#!/usr/bin/env bash
set -euo pipefail

# Submit Exp7 MLP-probe and Exp8 partial-finetuning runs on jchwang.
#
# Usage from node01:
#   bash scripts/run_probe_capacity_jch.sh /home/jchwang/ray/data/VinBigData_ChestXray
#
# Optional environment:
#   MODE=smoke|full
#   JOB_KIND=exp7|exp8|both
#   PBS_NODE=node02

DATA_DIR="${1:-${DATA_DIR:-}}"
if [ -z "$DATA_DIR" ]; then
  echo "Usage: bash scripts/run_probe_capacity_jch.sh /path/to/VinBigData_ChestXray" >&2
  exit 2
fi

ROOT=/home/jchwang/ray
REPO="$ROOT/JEPA"
VENV="${VENV:-$ROOT/venvs/mae-gpu}"
OUT_ROOT="${OUT_ROOT:-$REPO/results}"
JOB_DIR="${JOB_DIR:-$ROOT/jobs/probe_capacity}"
MODE="${MODE:-full}"
JOB_KIND="${JOB_KIND:-both}"
PBS_NODE="${PBS_NODE:-node02}"
DEVICE="${DEVICE:-cuda}"

IJEPA95="${IJEPA95:-$ROOT/pretrained/ijepa/jepa-ep95.pth.tar}"
IJEPA201="${IJEPA201:-$ROOT/pretrained/ijepa/jepa-ep201.pth.tar}"
MAE97="${MAE97:-$ROOT/outputs/mae_huge_mimic_97ep_imagenet_init_256_bs32/checkpoint-96.pth}"
MAE300="${MAE300:-$ROOT/outputs/mae_huge_mimic_300ep_imagenet_init_256_bs32/checkpoint-299.pth}"

mkdir -p "$JOB_DIR" "$OUT_ROOT"

if [ "$MODE" = "smoke" ]; then
  MAX_ARGS="--max_train_batches 2 --max_eval_batches 2"
  EXP7_EPOCHS=2
  EXP8_EPOCHS=1
  WALLTIME="02:00:00"
elif [ "$MODE" = "full" ]; then
  MAX_ARGS=""
  EXP7_EPOCHS="${EXP7_EPOCHS:-50}"
  EXP8_EPOCHS="${EXP8_EPOCHS:-15}"
  WALLTIME="${WALLTIME:-48:00:00}"
else
  echo "MODE must be smoke or full, got: $MODE" >&2
  exit 2
fi

cat > "$JOB_DIR/run_probe_capacity_${JOB_KIND}_${MODE}.pbs" <<EOF
#!/usr/bin/env bash
#PBS -N ray_probe_${JOB_KIND}_${MODE}
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

run_exp7 () {
  local model_type="\$1"
  local weights="\$2"
  local slug="\$3"
  python experiments/exp7_probe_capacity/run_exp7.py \\
    --model "\$model_type" \\
    --weights "\$weights" \\
    --data_dir "$DATA_DIR" \\
    --output_dir "$OUT_ROOT/exp7_mlp_probe_\$slug" \\
    --batch_size 32 \\
    --num_workers 8 \\
    --img_size 224 \\
    --in_chans 1 \\
    --device "$DEVICE" \\
    --probe_epochs "$EXP7_EPOCHS" \\
    $MAX_ARGS
}

run_exp8 () {
  local model_type="\$1"
  local weights="\$2"
  local slug="\$3"
  python experiments/exp8_partial_finetune/run_exp8.py \\
    --model "\$model_type" \\
    --weights "\$weights" \\
    --data_dir "$DATA_DIR" \\
    --output_dir "$OUT_ROOT/exp8_partial_ft_\$slug" \\
    --batch_size 16 \\
    --num_workers 8 \\
    --img_size 224 \\
    --in_chans 1 \\
    --device "$DEVICE" \\
    --epochs "$EXP8_EPOCHS" \\
    $MAX_ARGS
}

run_all_exp7 () {
  run_exp7 ijepa "$IJEPA95" "ijepa_vith_ep95_fixedsplit"
  run_exp7 ijepa "$IJEPA201" "ijepa_vith_ep201_fixedsplit"
  run_exp7 mae "$MAE97" "mae_huge_mimic_97ep_fixedsplit"
  run_exp7 mae "$MAE300" "mae_huge_mimic_300ep_fixedsplit"
}

run_all_exp8 () {
  run_exp8 ijepa "$IJEPA95" "ijepa_vith_ep95_fixedsplit"
  run_exp8 ijepa "$IJEPA201" "ijepa_vith_ep201_fixedsplit"
  run_exp8 mae "$MAE97" "mae_huge_mimic_97ep_fixedsplit"
  run_exp8 mae "$MAE300" "mae_huge_mimic_300ep_fixedsplit"
}

case "$JOB_KIND" in
  exp7) run_all_exp7 ;;
  exp8) run_all_exp8 ;;
  both) run_all_exp7; run_all_exp8 ;;
  *) echo "Invalid JOB_KIND=$JOB_KIND" >&2; exit 2 ;;
esac

python experiments/exp7_probe_capacity/plot_probe_capacity.py \\
  --linear_table "$REPO/results/paper_figures/table_exp1_robustness_long.csv" \\
  --exp7_inputs \\
    "I-JEPA-H/95=$OUT_ROOT/exp7_mlp_probe_ijepa_vith_ep95_fixedsplit" \\
    "I-JEPA-H/201=$OUT_ROOT/exp7_mlp_probe_ijepa_vith_ep201_fixedsplit" \\
    "MAE-H/97=$OUT_ROOT/exp7_mlp_probe_mae_huge_mimic_97ep_fixedsplit" \\
    "MAE-H/300=$OUT_ROOT/exp7_mlp_probe_mae_huge_mimic_300ep_fixedsplit" \\
  --exp8_inputs \\
    "I-JEPA-H/95=$OUT_ROOT/exp8_partial_ft_ijepa_vith_ep95_fixedsplit" \\
    "I-JEPA-H/201=$OUT_ROOT/exp8_partial_ft_ijepa_vith_ep201_fixedsplit" \\
    "MAE-H/97=$OUT_ROOT/exp8_partial_ft_mae_huge_mimic_97ep_fixedsplit" \\
    "MAE-H/300=$OUT_ROOT/exp8_partial_ft_mae_huge_mimic_300ep_fixedsplit" \\
  --output_dir "$REPO/results/paper_figures" || true

date
echo DONE_PROBE_CAPACITY_${JOB_KIND}_${MODE}
EOF

JOB_ID="$(qsub "$JOB_DIR/run_probe_capacity_${JOB_KIND}_${MODE}.pbs")"
echo "submitted=$JOB_ID"
echo "mode=$MODE"
echo "kind=$JOB_KIND"
echo "node=$PBS_NODE"
echo "log=$JOB_DIR/job_${JOB_KIND}_${MODE}.log"
