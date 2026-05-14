#!/usr/bin/env bash
set -euo pipefail

# Run MAE downstream experiments on the A800 server.
#
# Expected execution location: node01 login shell.
# The PBS jobs run on node07; paths are under /home/jchwang/ray.
#
# Usage:
#   bash scripts/run_mae_downstream_a800.sh /path/to/VinBigData_ChestXray
#
# Optional environment variables:
#   MAE_CKPT=/path/to/checkpoint-96.pth
#   OUT_ROOT=/home/jchwang/ray/JEPA/results
#   JOB_DIR=/home/jchwang/ray/jobs/mae_downstream

DATA_DIR="${1:-${DATA_DIR:-}}"
if [ -z "$DATA_DIR" ]; then
  echo "Usage: bash scripts/run_mae_downstream_a800.sh /path/to/VinBigData_ChestXray" >&2
  exit 2
fi

ROOT=/home/jchwang/ray
REPO="$ROOT/JEPA"
VENV="$ROOT/venvs/mae-gpu"
OUT_ROOT="${OUT_ROOT:-$REPO/results}"
JOB_DIR="${JOB_DIR:-$ROOT/jobs/mae_downstream}"

if [ -z "${MAE_CKPT:-}" ]; then
  CKPT_DIR="$ROOT/outputs/mae_huge_mimic_97ep_imagenet_init_256_bs32"
  MAE_CKPT="$(ls -1v "$CKPT_DIR"/checkpoint-*.pth 2>/dev/null | tail -1 || true)"
fi
if [ -z "$MAE_CKPT" ] || [ ! -f "$MAE_CKPT" ]; then
  echo "Could not find MAE checkpoint. Set MAE_CKPT=/path/to/checkpoint.pth" >&2
  exit 3
fi
if [ ! -d "$DATA_DIR" ]; then
  echo "DATA_DIR does not exist: $DATA_DIR" >&2
  exit 4
fi

mkdir -p "$JOB_DIR" "$OUT_ROOT"

EXP1_OUT="$OUT_ROOT/exp1_mae_huge_mimic_97ep"
EXP2_OUT="$OUT_ROOT/exp2_mae_huge_mimic_97ep"

cat > "$JOB_DIR/run_mae_downstream.pbs" <<EOF
#!/usr/bin/env bash
#PBS -N ray_mae_down
#PBS -q batch
#PBS -l nodes=node07:ppn=8
#PBS -l walltime=48:00:00

exec > $JOB_DIR/job.log 2>&1
set -euo pipefail
set -x

date
hostname
cd "$REPO"
. "$VENV/bin/activate"

echo "MAE_CKPT=$MAE_CKPT"
echo "DATA_DIR=$DATA_DIR"

python experiments/exp1_robustness/run_exp1.py \\
  --model mae \\
  --weights "$MAE_CKPT" \\
  --data_dir "$DATA_DIR" \\
  --output_dir "$EXP1_OUT" \\
  --batch_size 32 \\
  --num_workers 8 \\
  --img_size 224 \\
  --in_chans 1

python experiments/exp2_lesion_sensitivity/run_exp2.py \\
  --model mae \\
  --weights "$MAE_CKPT" \\
  --data_dir "$DATA_DIR" \\
  --probe_path "$EXP1_OUT/probe.pth" \\
  --output_dir "$EXP2_OUT" \\
  --batch_size 32 \\
  --num_workers 8 \\
  --img_size 224 \\
  --in_chans 1

python experiments/exp2_lesion_sensitivity/analyze_exp2.py \\
  --exp2_dir "$EXP2_OUT" \\
  --output_dir "$EXP2_OUT/analysis"

date
echo DONE
EOF

JOB_ID="$(qsub "$JOB_DIR/run_mae_downstream.pbs")"
echo "submitted=$JOB_ID"
echo "checkpoint=$MAE_CKPT"
echo "exp1_out=$EXP1_OUT"
echo "exp2_out=$EXP2_OUT"
echo "log=$JOB_DIR/job.log"
