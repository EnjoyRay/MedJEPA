#!/usr/bin/env bash
set -euo pipefail

# Submit Exp3/Exp4/Exp5 mechanism experiments to an A800 compute node.
#
# Expected execution location: node01 login shell.
# Usage:
#   bash scripts/run_mechanism_experiments_a800.sh /home/jchwang/ray/data/VinBigData_ChestXray
#
# Optional environment:
#   PBS_NODE=node08
#   VENV=/home/jchwang/ray/venvs/mae-gpu
#   OUT_ROOT=/home/jchwang/ray/JEPA/results
#   MAX_EXP4_SAMPLES=600
#   MAX_EXP3_BATCHES=0       # unset or 0 means full held-out split
#   EXP3_NOISE_REPEATS=5     # repeat stochastic Gaussian-noise controls
#   MAX_EXP5_TRAIN_BATCHES=0  # unset or 0 means full train split
#   MAX_EXP5_EVAL_BATCHES=0   # unset or 0 means full held-out split
#   DEVICE=cuda               # use cpu only for smoke tests on non-supported GPUs
#   MECH_BATCH_SIZE=16         # lower to 8 or 4 on smaller GPUs if needed

DATA_DIR="${1:-${DATA_DIR:-}}"
if [ -z "$DATA_DIR" ]; then
  echo "Usage: bash scripts/run_mechanism_experiments_a800.sh /path/to/VinBigData_ChestXray" >&2
  exit 2
fi

ROOT=/home/jchwang/ray
REPO="$ROOT/JEPA"
VENV="${VENV:-$ROOT/venvs/mae-gpu}"
OUT_ROOT="${OUT_ROOT:-$REPO/results}"
JOB_DIR="${JOB_DIR:-$ROOT/jobs/mechanism_experiments}"
PBS_NODE="${PBS_NODE:-node08}"
DEVICE="${DEVICE:-cuda}"
MECH_BATCH_SIZE="${MECH_BATCH_SIZE:-16}"
EXP3_NOISE_REPEATS="${EXP3_NOISE_REPEATS:-5}"

IJEPA95="${IJEPA95:-$ROOT/pretrained/ijepa/jepa-ep95.pth.tar}"
IJEPA201="${IJEPA201:-$ROOT/pretrained/ijepa/jepa-ep201.pth.tar}"
MAE97="${MAE97:-$ROOT/outputs/mae_huge_mimic_97ep_imagenet_init_256_bs32/checkpoint-96.pth}"

PROBE_IJEPA95="${PROBE_IJEPA95:-$OUT_ROOT/exp1_ijepa_vith_ep95_fixedsplit/probe.pth}"
PROBE_IJEPA201="${PROBE_IJEPA201:-$OUT_ROOT/exp1_ijepa_vith_ep201_fixedsplit/probe.pth}"
PROBE_MAE97="${PROBE_MAE97:-$OUT_ROOT/exp1_mae_huge_mimic_97ep_fixedsplit/probe.pth}"

mkdir -p "$JOB_DIR" "$OUT_ROOT"

if [ "${MAX_EXP5_TRAIN_BATCHES:-0}" = "0" ]; then
  EXP5_TRAIN_ARGS=""
else
  EXP5_TRAIN_ARGS="--max_train_batches ${MAX_EXP5_TRAIN_BATCHES}"
fi
if [ "${MAX_EXP3_BATCHES:-0}" = "0" ]; then
  EXP3_ARGS=""
else
  EXP3_ARGS="--max_batches ${MAX_EXP3_BATCHES}"
fi
if [ "${MAX_EXP5_EVAL_BATCHES:-0}" = "0" ]; then
  EXP5_EVAL_ARGS=""
else
  EXP5_EVAL_ARGS="--max_eval_batches ${MAX_EXP5_EVAL_BATCHES}"
fi

cat > "$JOB_DIR/run_mechanism_experiments.pbs" <<EOF
#!/usr/bin/env bash
#PBS -N ray_mech_exp
#PBS -q batch
#PBS -l nodes=${PBS_NODE}:ppn=8
#PBS -l walltime=72:00:00

exec > $JOB_DIR/job.log 2>&1
set -euo pipefail
set -x

date
hostname
cd "$REPO"
. "$VENV/bin/activate"

run_model () {
  local model_name="\$1"
  local model_type="\$2"
  local weights="\$3"
  local probe="\$4"
  local slug="\$5"

  test -f "\$weights"
  test -f "\$probe"

  python experiments/exp3_frequency_sensitivity/run_exp3.py --model "\$model_type" --weights "\$weights" --data_dir "$DATA_DIR" --probe_path "\$probe" --output_dir "$OUT_ROOT/exp3_frequency_\$slug" --batch_size "$MECH_BATCH_SIZE" --num_workers 8 --img_size 224 --in_chans 1 --device "$DEVICE" --bootstrap_iters 300 --noise_repeats "$EXP3_NOISE_REPEATS" $EXP3_ARGS

  python experiments/exp4_mechanism/run_exp4.py \\
    --model "\$model_type" \\
    --weights "\$weights" \\
    --data_dir "$DATA_DIR" \\
    --probe_path "\$probe" \\
    --output_dir "$OUT_ROOT/exp4_mechanism_\$slug" \\
    --batch_size 1 \\
    --num_workers 4 \\
    --img_size 224 \\
    --in_chans 1 \\
    --device "$DEVICE" \\
    --max_samples "${MAX_EXP4_SAMPLES:-600}"

  python experiments/exp5_lightweight_mitigation/run_exp5.py --model "\$model_type" --weights "\$weights" --data_dir "$DATA_DIR" --probe_path "\$probe" --output_dir "$OUT_ROOT/exp5_mitigation_\$slug" --batch_size "$MECH_BATCH_SIZE" --num_workers 8 --img_size 224 --in_chans 1 --device "$DEVICE" --probe_epochs 30 $EXP5_TRAIN_ARGS $EXP5_EVAL_ARGS
}

run_model "I-JEPA-H/95" ijepa "$IJEPA95" "$PROBE_IJEPA95" "ijepa_vith_ep95_fixedsplit"
run_model "I-JEPA-H/201" ijepa "$IJEPA201" "$PROBE_IJEPA201" "ijepa_vith_ep201_fixedsplit"
run_model "MAE-H/97" mae "$MAE97" "$PROBE_MAE97" "mae_huge_mimic_97ep_fixedsplit"

python experiments/exp3_frequency_sensitivity/plot_exp3.py \\
  --inputs \\
    "I-JEPA-H/95=$OUT_ROOT/exp3_frequency_ijepa_vith_ep95_fixedsplit/frequency_sensitivity.csv" \\
    "I-JEPA-H/201=$OUT_ROOT/exp3_frequency_ijepa_vith_ep201_fixedsplit/frequency_sensitivity.csv" \\
    "MAE-H/97=$OUT_ROOT/exp3_frequency_mae_huge_mimic_97ep_fixedsplit/frequency_sensitivity.csv" \\
  --output_dir "$REPO/results/paper_figures"

python experiments/exp4_mechanism/plot_exp4.py \\
  --inputs \\
    "I-JEPA-H/95=$OUT_ROOT/exp4_mechanism_ijepa_vith_ep95_fixedsplit/mechanism_per_sample.csv" \\
    "I-JEPA-H/201=$OUT_ROOT/exp4_mechanism_ijepa_vith_ep201_fixedsplit/mechanism_per_sample.csv" \\
    "MAE-H/97=$OUT_ROOT/exp4_mechanism_mae_huge_mimic_97ep_fixedsplit/mechanism_per_sample.csv" \\
  --output_dir "$REPO/results/paper_figures"

python experiments/exp5_lightweight_mitigation/plot_exp5.py \\
  --inputs \\
    "I-JEPA-H/95=$OUT_ROOT/exp5_mitigation_ijepa_vith_ep95_fixedsplit/mitigation_results.csv" \\
    "I-JEPA-H/201=$OUT_ROOT/exp5_mitigation_ijepa_vith_ep201_fixedsplit/mitigation_results.csv" \\
    "MAE-H/97=$OUT_ROOT/exp5_mitigation_mae_huge_mimic_97ep_fixedsplit/mitigation_results.csv" \\
  --output_dir "$REPO/results/paper_figures"

date
echo DONE_MECHANISM_EXPERIMENTS
EOF

JOB_ID="$(qsub "$JOB_DIR/run_mechanism_experiments.pbs")"
echo "submitted=$JOB_ID"
echo "node=$PBS_NODE"
echo "device=$DEVICE"
echo "log=$JOB_DIR/job.log"
