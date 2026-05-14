#!/usr/bin/env bash
set -euo pipefail

# Submit Exp6 Noise-Consistent Adapter runs to a jchwang compute node.
#
# Expected execution location: node01 login shell.
# Usage:
#   bash scripts/run_exp6_nca_jch.sh /home/jchwang/ray/data/VinBigData_ChestXray
#
# Optional environment:
#   PBS_NODE=node02
#   VENV=/home/jchwang/ray/venvs/mae-gpu
#   MODE=smoke|full
#   DEVICE=cuda
#   EXP6_BATCH_SIZE=16
#   EXP6_ADAPTER_EPOCHS=30
#   EXP6_ADAPTER_BATCH_SIZE=256

DATA_DIR="${1:-${DATA_DIR:-}}"
if [ -z "$DATA_DIR" ]; then
  echo "Usage: bash scripts/run_exp6_nca_jch.sh /path/to/VinBigData_ChestXray" >&2
  exit 2
fi

ROOT=/home/jchwang/ray
REPO="$ROOT/JEPA"
VENV="${VENV:-$ROOT/venvs/mae-gpu}"
OUT_ROOT="${OUT_ROOT:-$REPO/results}"
JOB_DIR="${JOB_DIR:-$ROOT/jobs/exp6_nca}"
PBS_NODE="${PBS_NODE:-node02}"
MODE="${MODE:-full}"
DEVICE="${DEVICE:-cuda}"
EXP6_BATCH_SIZE="${EXP6_BATCH_SIZE:-16}"
EXP6_ADAPTER_EPOCHS="${EXP6_ADAPTER_EPOCHS:-30}"
EXP6_ADAPTER_BATCH_SIZE="${EXP6_ADAPTER_BATCH_SIZE:-256}"

IJEPA95="${IJEPA95:-$ROOT/pretrained/ijepa/jepa-ep95.pth.tar}"
IJEPA201="${IJEPA201:-$ROOT/pretrained/ijepa/jepa-ep201.pth.tar}"
MAE97="${MAE97:-$ROOT/outputs/mae_huge_mimic_97ep_imagenet_init_256_bs32/checkpoint-96.pth}"
MAE300="${MAE300:-$ROOT/outputs/mae_huge_mimic_300ep_imagenet_init_256_bs32/checkpoint-299.pth}"

mkdir -p "$JOB_DIR" "$OUT_ROOT"

if [ "$MODE" = "smoke" ]; then
  MAX_ARGS="--max_train_batches 2 --max_eval_batches 2"
  ABLATIONS_IJEPA201="full"
  ABLATIONS_OTHER="full"
  EPOCHS=2
elif [ "$MODE" = "full" ]; then
  MAX_ARGS=""
  ABLATIONS_IJEPA201="all"
  ABLATIONS_OTHER="full"
  EPOCHS="$EXP6_ADAPTER_EPOCHS"
else
  echo "MODE must be smoke or full, got: $MODE" >&2
  exit 2
fi

cat > "$JOB_DIR/run_exp6_nca_${MODE}.pbs" <<EOF
#!/usr/bin/env bash
#PBS -N ray_exp6_nca_${MODE}
#PBS -q batch
#PBS -l nodes=${PBS_NODE}:ppn=8
#PBS -l walltime=96:00:00

exec > $JOB_DIR/job_${MODE}.log 2>&1
set -euo pipefail
set -x

date
hostname
nvidia-smi || true
cd "$REPO"
. "$VENV/bin/activate"

run_nca () {
  local model_type="\$1"
  local weights="\$2"
  local slug="\$3"
  local ablations="\$4"

  test -f "\$weights"
  python experiments/exp6_noise_consistent_adapter/run_exp6.py \\
    --model "\$model_type" \\
    --weights "\$weights" \\
    --data_dir "$DATA_DIR" \\
    --output_dir "$OUT_ROOT/exp6_nca_\$slug" \\
    --batch_size "$EXP6_BATCH_SIZE" \\
    --num_workers 8 \\
    --img_size 224 \\
    --in_chans 1 \\
    --device "$DEVICE" \\
    --adapter_epochs "$EPOCHS" \\
    --adapter_batch_size "$EXP6_ADAPTER_BATCH_SIZE" \\
    --ablations "\$ablations" \\
    $MAX_ARGS
}

run_nca ijepa "$IJEPA201" "ijepa_vith_ep201_fixedsplit" "$ABLATIONS_IJEPA201"

if [ "$MODE" = "full" ]; then
  run_nca ijepa "$IJEPA95" "ijepa_vith_ep95_fixedsplit" "$ABLATIONS_OTHER"
  run_nca mae "$MAE97" "mae_huge_mimic_97ep_fixedsplit" "$ABLATIONS_OTHER"
  run_nca mae "$MAE300" "mae_huge_mimic_300ep_fixedsplit" "$ABLATIONS_OTHER"
fi

python experiments/exp6_noise_consistent_adapter/plot_exp6.py \\
  --inputs \\
    "I-JEPA-H/95=$OUT_ROOT/exp6_nca_ijepa_vith_ep95_fixedsplit" \\
    "I-JEPA-H/201=$OUT_ROOT/exp6_nca_ijepa_vith_ep201_fixedsplit" \\
    "MAE-H/97=$OUT_ROOT/exp6_nca_mae_huge_mimic_97ep_fixedsplit" \\
    "MAE-H/300=$OUT_ROOT/exp6_nca_mae_huge_mimic_300ep_fixedsplit" \\
  --exp5_inputs \\
    "I-JEPA-H/95=$OUT_ROOT/exp5_mitigation_ijepa_vith_ep95_fixedsplit" \\
    "I-JEPA-H/201=$OUT_ROOT/exp5_mitigation_ijepa_vith_ep201_fixedsplit" \\
    "MAE-H/97=$OUT_ROOT/exp5_mitigation_mae_huge_mimic_97ep_fixedsplit" \\
    "MAE-H/300=$OUT_ROOT/exp5_mitigation_mae_huge_mimic_300ep_fixedsplit" \\
  --output_dir "$REPO/results/paper_figures"

date
echo DONE_EXP6_NCA_${MODE}
EOF

JOB_ID="$(qsub "$JOB_DIR/run_exp6_nca_${MODE}.pbs")"
echo "submitted=$JOB_ID"
echo "mode=$MODE"
echo "node=$PBS_NODE"
echo "log=$JOB_DIR/job_${MODE}.log"
