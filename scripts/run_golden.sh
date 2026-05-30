#!/usr/bin/env bash
set -euo pipefail
cd /home/uic2/Raymond
source /home/uic2/miniconda3/etc/profile.d/conda.sh
conda activate medical_ijepa
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

DATA_DIR=/home/uic2/zhaoyi/VinBigData_ChestXray
RESULTS=/home/uic2/Raymond/results
mkdir -p ${RESULTS}

CKPT_BASE=/home/uic2/zhaoyi/medical-i-jepa/golden_checkpoints

# Map: short_name -> checkpoint_filename
declare -A CKPT_MAP
CKPT_MAP[v3_1_ep201]=v3.1_ep201_jepa.pth.tar
CKPT_MAP[v4_ep50]=v4_ep50_jepa.pth.tar
CKPT_MAP[v5_ep60]=v5_ep60_jepa.pth.tar
CKPT_MAP[v6_ep40]=v6_ep40_jepa.pth.tar

echo "Start: $(date)"

# ===== Exp2b: class-aligned =====
for NAME in v3_1_ep201 v4_ep50 v5_ep60 v6_ep40; do
  OUT=${RESULTS}/exp2b_ijepa_vith_${NAME}_fixedsplit
  CKPT=${CKPT_BASE}/${CKPT_MAP[${NAME}]}
  mkdir -p ${OUT}
  echo "=== Exp2b ${NAME} $(date) ==="
  python -u experiments/exp2_lesion_sensitivity/run_exp2_class_aligned.py \
    --model ijepa --weights ${CKPT} --data_dir ${DATA_DIR} \
    --output_dir ${OUT} --num_controls 5 --control_strategy matched_random \
    --max_samples 999999 --batch_size 8 --encoder_batch_size 8 \
    2>&1 | tee ${OUT}/run.log
  echo "Exp2b ${NAME} done: $(date)"
done

# ===== Exp3: frequency =====
for NAME in v3_1_ep201 v4_ep50 v5_ep60 v6_ep40; do
  OUT=${RESULTS}/exp3_frequency_ijepa_vith_${NAME}_fixedsplit
  PROBE=${RESULTS}/exp2b_ijepa_vith_${NAME}_fixedsplit/probe.pth
  CKPT=${CKPT_BASE}/${CKPT_MAP[${NAME}]}
  mkdir -p ${OUT}
  echo "=== Exp3 ${NAME} $(date) ==="
  python -u experiments/exp3_frequency_sensitivity/run_exp3.py \
    --model ijepa --weights ${CKPT} --data_dir ${DATA_DIR} \
    --probe_path ${PROBE} --output_dir ${OUT} \
    2>&1 | tee ${OUT}/run.log
  echo "Exp3 ${NAME} done: $(date)"
done

# ===== Exp7: MLP probe (skip v3.1=ep201) =====
for NAME in v4_ep50 v5_ep60 v6_ep40; do
  OUT=${RESULTS}/exp7_mlp_probe_ijepa_vith_${NAME}_fixedsplit
  CKPT=${CKPT_BASE}/${CKPT_MAP[${NAME}]}
  mkdir -p ${OUT}
  echo "=== Exp7 ${NAME} $(date) ==="
  python -u experiments/exp7_probe_capacity/run_exp7.py \
    --model ijepa --weights ${CKPT} --data_dir ${DATA_DIR} \
    --output_dir ${OUT} \
    2>&1 | tee ${OUT}/run.log
  echo "Exp7 ${NAME} done: $(date)"
done

# ===== Exp8: partial FT (skip v3.1=ep201) =====
for NAME in v4_ep50 v5_ep60 v6_ep40; do
  OUT=${RESULTS}/exp8_partial_ft_ijepa_vith_${NAME}_fixedsplit
  CKPT=${CKPT_BASE}/${CKPT_MAP[${NAME}]}
  mkdir -p ${OUT}
  echo "=== Exp8 ${NAME} $(date) ==="
  python -u experiments/exp8_partial_finetune/run_exp8.py \
    --model ijepa --weights ${CKPT} --data_dir ${DATA_DIR} \
    --output_dir ${OUT} \
    2>&1 | tee ${OUT}/run.log
  echo "Exp8 ${NAME} done: $(date)"
done

echo "ALL DONE: $(date)"
