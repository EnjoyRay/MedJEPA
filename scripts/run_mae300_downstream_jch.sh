#!/usr/bin/env bash
set -euo pipefail

# Submit MAE-H/300 downstream and mechanism experiments on the jch PBS cluster.
#
# Expected execution location: node01.
# Usage:
#   PBS_NODE=node02 bash scripts/run_mae300_downstream_jch.sh /home/jchwang/ray/data/VinBigData_ChestXray
#
# Optional environment:
#   PBS_NODE=node02
#   DEVICE=cuda
#   MECH_BATCH_SIZE=8
#   MAX_EXP1_TRAIN_BATCHES=0  # 0 means full
#   MAX_EXP1_EVAL_BATCHES=0   # 0 means full
#   MAX_EXP2B_SAMPLES=0       # 0 means full
#   MAX_EXP3_BATCHES=0        # 0 means full
#   EXP3_NOISE_REPEATS=5      # repeat stochastic Gaussian-noise controls
#   MAX_EXP4_SAMPLES=600
#   MAX_EXP5_TRAIN_BATCHES=0  # 0 means full
#   MAX_EXP5_EVAL_BATCHES=0   # 0 means full

DATA_DIR="${1:-${DATA_DIR:-}}"
if [ -z "$DATA_DIR" ]; then
  echo "Usage: bash scripts/run_mae300_downstream_jch.sh /path/to/VinBigData_ChestXray" >&2
  exit 2
fi

ROOT=/home/jchwang/ray
REPO="$ROOT/JEPA"
VENV="${VENV:-$ROOT/venvs/mae-gpu}"
OUT_ROOT="${OUT_ROOT:-$REPO/results}"
JOB_DIR="${JOB_DIR:-$ROOT/jobs/mae300_downstream}"
PBS_NODE="${PBS_NODE:-node02}"
DEVICE="${DEVICE:-cuda}"
MECH_BATCH_SIZE="${MECH_BATCH_SIZE:-8}"
EXP3_NOISE_REPEATS="${EXP3_NOISE_REPEATS:-5}"
MAE300="${MAE300:-$ROOT/outputs/mae_huge_mimic_300ep_imagenet_init_256_bs32/checkpoint-299.pth}"

EXP1_OUT="$OUT_ROOT/exp1_mae_huge_mimic_300ep_fixedsplit"
EXP2B_OUT="$OUT_ROOT/exp2b_class_aligned_mae_huge_mimic_300ep_fixedsplit"
EXP3_OUT="$OUT_ROOT/exp3_frequency_mae_huge_mimic_300ep_fixedsplit"
EXP4_OUT="$OUT_ROOT/exp4_mechanism_mae_huge_mimic_300ep_fixedsplit"
EXP5_OUT="$OUT_ROOT/exp5_mitigation_mae_huge_mimic_300ep_fixedsplit"

test -f "$MAE300"
test -d "$DATA_DIR"
mkdir -p "$JOB_DIR" "$OUT_ROOT"

if [ "${MAX_EXP2B_SAMPLES:-0}" = "0" ]; then EXP2B_ARGS=""; else EXP2B_ARGS="--max_samples ${MAX_EXP2B_SAMPLES}"; fi
if [ "${MAX_EXP3_BATCHES:-0}" = "0" ]; then EXP3_ARGS=""; else EXP3_ARGS="--max_batches ${MAX_EXP3_BATCHES}"; fi
if [ "${MAX_EXP1_TRAIN_BATCHES:-0}" = "0" ]; then EXP1_TRAIN_ARGS=""; else EXP1_TRAIN_ARGS="--max_train_batches ${MAX_EXP1_TRAIN_BATCHES}"; fi
if [ "${MAX_EXP1_EVAL_BATCHES:-0}" = "0" ]; then EXP1_EVAL_ARGS=""; else EXP1_EVAL_ARGS="--max_eval_batches ${MAX_EXP1_EVAL_BATCHES}"; fi
if [ "${MAX_EXP5_TRAIN_BATCHES:-0}" = "0" ]; then EXP5_TRAIN_ARGS=""; else EXP5_TRAIN_ARGS="--max_train_batches ${MAX_EXP5_TRAIN_BATCHES}"; fi
if [ "${MAX_EXP5_EVAL_BATCHES:-0}" = "0" ]; then EXP5_EVAL_ARGS=""; else EXP5_EVAL_ARGS="--max_eval_batches ${MAX_EXP5_EVAL_BATCHES}"; fi

cat > "$JOB_DIR/run_mae300_downstream.pbs" <<EOF
#!/usr/bin/env bash
#PBS -N ray_mae300_ds
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

echo "MAE300=$MAE300"
echo "DATA_DIR=$DATA_DIR"

python experiments/exp1_robustness/run_exp1.py --model mae --weights "$MAE300" --data_dir "$DATA_DIR" --output_dir "$EXP1_OUT" --batch_size "$MECH_BATCH_SIZE" --num_workers 8 --img_size 224 --in_chans 1 --device "$DEVICE" $EXP1_TRAIN_ARGS $EXP1_EVAL_ARGS

python experiments/exp2_lesion_sensitivity/run_exp2_class_aligned.py \\
  --model mae \\
  --weights "$MAE300" \\
  --data_dir "$DATA_DIR" \\
  --probe_path "$EXP1_OUT/probe.pth" \\
  --output_dir "$EXP2B_OUT" \\
  --batch_size "$MECH_BATCH_SIZE" \\
  --num_workers 8 \\
  --img_size 224 \\
  --in_chans 1 \\
  --device "$DEVICE" \\
  --encoder_batch_size "$MECH_BATCH_SIZE" \\
  $EXP2B_ARGS

python experiments/exp3_frequency_sensitivity/run_exp3.py --model mae --weights "$MAE300" --data_dir "$DATA_DIR" --probe_path "$EXP1_OUT/probe.pth" --output_dir "$EXP3_OUT" --batch_size "$MECH_BATCH_SIZE" --num_workers 8 --img_size 224 --in_chans 1 --device "$DEVICE" --bootstrap_iters 300 --noise_repeats "$EXP3_NOISE_REPEATS" $EXP3_ARGS

python experiments/exp4_mechanism/run_exp4.py \\
  --model mae \\
  --weights "$MAE300" \\
  --data_dir "$DATA_DIR" \\
  --probe_path "$EXP1_OUT/probe.pth" \\
  --output_dir "$EXP4_OUT" \\
  --batch_size 1 \\
  --num_workers 4 \\
  --img_size 224 \\
  --in_chans 1 \\
  --device "$DEVICE" \\
  --max_samples "${MAX_EXP4_SAMPLES:-600}"

python experiments/exp5_lightweight_mitigation/run_exp5.py --model mae --weights "$MAE300" --data_dir "$DATA_DIR" --probe_path "$EXP1_OUT/probe.pth" --output_dir "$EXP5_OUT" --batch_size "$MECH_BATCH_SIZE" --num_workers 8 --img_size 224 --in_chans 1 --device "$DEVICE" --probe_epochs 30 $EXP5_TRAIN_ARGS $EXP5_EVAL_ARGS

python experiments/exp3_frequency_sensitivity/plot_exp3.py \\
  --inputs \\
    "I-JEPA-H/95=$OUT_ROOT/exp3_frequency_ijepa_vith_ep95_fixedsplit/frequency_sensitivity.csv" \\
    "I-JEPA-H/201=$OUT_ROOT/exp3_frequency_ijepa_vith_ep201_fixedsplit/frequency_sensitivity.csv" \\
    "MAE-H/97=$OUT_ROOT/exp3_frequency_mae_huge_mimic_97ep_fixedsplit/frequency_sensitivity.csv" \\
    "MAE-H/300=$EXP3_OUT/frequency_sensitivity.csv" \\
  --output_dir "$REPO/results/paper_figures"

python experiments/exp4_mechanism/plot_exp4.py \\
  --inputs \\
    "I-JEPA-H/95=$OUT_ROOT/exp4_mechanism_ijepa_vith_ep95_fixedsplit/mechanism_per_sample.csv" \\
    "I-JEPA-H/201=$OUT_ROOT/exp4_mechanism_ijepa_vith_ep201_fixedsplit/mechanism_per_sample.csv" \\
    "MAE-H/97=$OUT_ROOT/exp4_mechanism_mae_huge_mimic_97ep_fixedsplit/mechanism_per_sample.csv" \\
    "MAE-H/300=$EXP4_OUT/mechanism_per_sample.csv" \\
  --output_dir "$REPO/results/paper_figures"

python experiments/exp5_lightweight_mitigation/plot_exp5.py \\
  --inputs \\
    "I-JEPA-H/95=$OUT_ROOT/exp5_mitigation_ijepa_vith_ep95_fixedsplit/mitigation_results.csv" \\
    "I-JEPA-H/201=$OUT_ROOT/exp5_mitigation_ijepa_vith_ep201_fixedsplit/mitigation_results.csv" \\
    "MAE-H/97=$OUT_ROOT/exp5_mitigation_mae_huge_mimic_97ep_fixedsplit/mitigation_results.csv" \\
    "MAE-H/300=$EXP5_OUT/mitigation_results.csv" \\
  --output_dir "$REPO/results/paper_figures"

date
echo DONE_MAE300_DOWNSTREAM
EOF

JOB_ID="$(qsub "$JOB_DIR/run_mae300_downstream.pbs")"
echo "submitted=$JOB_ID"
echo "node=$PBS_NODE"
echo "checkpoint=$MAE300"
echo "exp1_out=$EXP1_OUT"
echo "log=$JOB_DIR/job.log"
