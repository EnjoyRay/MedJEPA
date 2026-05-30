#!/usr/bin/env bash
set -euo pipefail

# Submit HPM pretraining on the jchwang PBS cluster.
#
# Expected execution location: node01/login shell on jchwang.
# Usage:
#   EPOCHS=300 PBS_NODE=node02 bash scripts/run_hpm_pretrain_jch.sh /home/jchwang/ray/data/MIMIC_CXR
#
# The script creates the ImageListFolder train.txt expected by HPM. Labels are
# dummy zeros because HPM pretraining is self-supervised.

DATA_DIR="${1:-${DATA_DIR:-}}"
if [ -z "$DATA_DIR" ]; then
  echo "Usage: EPOCHS=300 bash scripts/run_hpm_pretrain_jch.sh /path/to/mimic_data_root" >&2
  exit 2
fi

ROOT="${ROOT:-/home/jchwang/ray}"
HPM_REPO="${HPM_REPO:-$ROOT/HPM}"
VENV="${VENV:-$ROOT/venvs/mae-gpu}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs}"
JOB_DIR="${JOB_DIR:-$ROOT/jobs/hpm_pretrain}"
DATA_INDEX_ROOT="${DATA_INDEX_ROOT:-$ROOT/data/hpm_mimic_index}"
PBS_NODE="${PBS_NODE:-node02}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
EPOCHS="${EPOCHS:-300}"
WALLTIME="${WALLTIME:-168:00:00}"

# MAE-H on this project used batch_size=32, accum_iter=8 on one GPU, i.e.
# effective batch 256. On 4x 5090, batch_size=2 + accum_iter=32 matches that.
BATCH_SIZE="${BATCH_SIZE:-2}"
ACCUM_ITER="${ACCUM_ITER:-32}"
MODEL="${MODEL:-mae_vit_huge_patch14_dec512d8b}"
INPUT_SIZE="${INPUT_SIZE:-224}"
TOKEN_SIZE="${TOKEN_SIZE:-16}"
MASK_RATIO="${MASK_RATIO:-0.75}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"
BLR="${BLR:-1.5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
NUM_WORKERS="${NUM_WORKERS:-8}"
INIT_CKPT="${INIT_CKPT:-}"
EXPERIMENT="${EXPERIMENT:-hpm_vith14_mimic_ep${EPOCHS}_bs${BATCH_SIZE}x${NPROC_PER_NODE}_acc${ACCUM_ITER}}"
OUTPUT_DIR="${OUTPUT_DIR:-$OUT_ROOT/$EXPERIMENT}"
LOG_DIR="${LOG_DIR:-$ROOT/logs/hpm_pretrain}"

mkdir -p "$JOB_DIR" "$OUT_ROOT" "$LOG_DIR"

cat > "$JOB_DIR/run_${EXPERIMENT}.pbs" <<EOF
#!/usr/bin/env bash
#PBS -N ray_hpm_${EPOCHS}
#PBS -q batch
#PBS -l nodes=${PBS_NODE}:ppn=8
#PBS -l walltime=${WALLTIME}

exec > $JOB_DIR/job_${EXPERIMENT}.log 2>&1
set -euo pipefail
set -x

date
hostname
nvidia-smi || true

if [ ! -d "$HPM_REPO" ]; then
  git clone https://github.com/Eric-lfmself/HPM.git "$HPM_REPO"
elif [ -d "$HPM_REPO/.git" ]; then
  git -C "$HPM_REPO" pull --ff-only || true
fi

cd "$HPM_REPO"

# Minimal compatibility patch for this fork:
# 1. args.byol is referenced but not defined.
# 2. timm version assertion is too strict for the existing mae-gpu env.
# 3. timm.optim.optim_factory.add_weight_decay was renamed in newer timm.
# 4. args.load_from is parsed but unused; wire it to partial weight loading.
python - <<'PY'
import re
from pathlib import Path

p = Path("main_pretrain.py")
s = p.read_text()
s = s.replace('assert timm.__version__ == "0.3.2"  ', '# patched: allow the server timm version')
if "--byol" not in s:
    s = s.replace(
        "parser.add_argument('--bf16', action='store_true', help='whether to use bf16')",
        "parser.add_argument('--bf16', action='store_true', help='whether to use bf16')\n"
        "    parser.add_argument('--byol', action='store_true', help='compatibility flag; disabled by default')",
    )
s = re.sub(
    r"    # following timm: set wd as 0 for bias and norm layers\n.*?    optimizer = torch\.optim\.AdamW",
    "    # following timm: set wd as 0 for bias and norm layers\n"
    "    if hasattr(optim_factory, 'add_weight_decay'):\n"
    "        param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)\n"
    "    else:\n"
    "        param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)\n"
    "    optimizer = torch.optim.AdamW",
    s,
    flags=re.S,
)
if "Patched init load from args.load_from" not in s:
    marker = '    print("Model = %s" % str(model_without_ddp))\n'
    init_block = '''    print("Model = %s" % str(model_without_ddp))

    # Patched init load from args.load_from. HPM has extra loss-prediction
    # modules, so load only checkpoint tensors whose names and shapes match.
    if args.load_from:
        print("Loading init checkpoint from %s" % args.load_from)
        checkpoint = torch.load(args.load_from, map_location='cpu')
        checkpoint_model = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        model_state = model_without_ddp.state_dict()
        loaded_state = {}
        skipped = []
        for k, v in checkpoint_model.items():
            name = k[7:] if k.startswith('module.') else k
            if name in model_state and tuple(v.shape) == tuple(model_state[name].shape):
                loaded_state[name] = v
            else:
                skipped.append(name)
        msg = model_without_ddp.load_state_dict(loaded_state, strict=False)
        print("Loaded %d tensors from init checkpoint; skipped %d tensors" % (len(loaded_state), len(skipped)))
        print(msg)
'''
    s = s.replace(marker, init_block)
p.write_text(s)

for model_path in ["models_mae.py", "models_mae_learn_loss.py", "models_mae_learn_feature_loss.py"]:
    q = Path(model_path)
    t = q.read_text()
    t = t.replace(", qk_scale=None", "")
    q.write_text(t)

q = Path("util/pos_embed.py")
t = q.read_text()
t = t.replace("dtype=float32", "dtype=np.float32")
t = t.replace("dtype=np.float)", "dtype=float)")
t = t.replace("dtype=np.float,", "dtype=float,")
q.write_text(t)
PY

. "$VENV/bin/activate"

test -d "$DATA_DIR"
mkdir -p "$DATA_INDEX_ROOT"
rm -f "$DATA_INDEX_ROOT/train"
ln -s "$DATA_DIR" "$DATA_INDEX_ROOT/train"

if [ -f "$DATA_DIR/train_images.txt" ]; then
  DATA_DIR_FOR_INDEX="$DATA_DIR" DATA_INDEX_ROOT_FOR_INDEX="$DATA_INDEX_ROOT" python - <<'PY'
import os
from pathlib import Path

data_dir = Path(os.environ["DATA_DIR_FOR_INDEX"]).absolute()
out = Path(os.environ["DATA_INDEX_ROOT_FOR_INDEX"]) / "train.txt"
src = data_dir / "train_images.txt"
rows = []
for line in src.read_text().splitlines():
    if not line.strip():
        continue
    path = Path(line.strip())
    if path.is_absolute():
        try:
            rel = path.relative_to(data_dir)
        except ValueError:
            rel = Path(*path.parts[-6:])
    else:
        rel = path
    rows.append(f"{rel.as_posix()} 0")
out.write_text("\n".join(rows) + "\n")
PY
elif [ "$DATA_DIR" = "$ROOT/data/mimic_cxr_imagefolder_256" ] && [ -f "$ROOT/data/mimic_cxr_imagefolder/train_images.txt" ]; then
  ROOT_FOR_INDEX="$ROOT" DATA_INDEX_ROOT_FOR_INDEX="$DATA_INDEX_ROOT" python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["ROOT_FOR_INDEX"])
src_root = root / "data" / "mimic_cxr_imagefolder"
src = src_root / "train_images.txt"
out = Path(os.environ["DATA_INDEX_ROOT_FOR_INDEX"]) / "train.txt"
rows = []
for line in src.read_text().splitlines():
    if not line.strip():
        continue
    path = Path(line.strip())
    rel = path.relative_to(src_root) if path.is_absolute() else path
    rows.append(f"{rel.as_posix()} 0")
out.write_text("\n".join(rows) + "\n")
PY
else
  DATA_DIR_FOR_INDEX="$DATA_DIR" DATA_INDEX_ROOT_FOR_INDEX="$DATA_INDEX_ROOT" python - <<'PY'
import os
from pathlib import Path

data_dir = Path(os.environ["DATA_DIR_FOR_INDEX"]).absolute()
out = Path(os.environ["DATA_INDEX_ROOT_FOR_INDEX"]) / "train.txt"
suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
rows = []
for path in data_dir.rglob("*"):
    if path.is_file() and path.suffix.lower() in suffixes:
        rows.append(f"{path.relative_to(data_dir).as_posix()} 0")
rows.sort()
out.write_text("\n".join(rows) + "\n")
PY
fi

test -s "$DATA_INDEX_ROOT/train.txt"
head "$DATA_INDEX_ROOT/train.txt"
wc -l "$DATA_INDEX_ROOT/train.txt"

mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=()
if [ -n "$INIT_CKPT" ]; then
  test -f "$INIT_CKPT"
  EXTRA_ARGS+=(--load_from "$INIT_CKPT")
fi

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" main_pretrain.py \\
  --batch_size "$BATCH_SIZE" \\
  --accum_iter "$ACCUM_ITER" \\
  --model "$MODEL" \\
  --input_size "$INPUT_SIZE" \\
  --token_size "$TOKEN_SIZE" \\
  --mask_ratio "$MASK_RATIO" \\
  --epochs "$EPOCHS" \\
  --warmup_epochs "$WARMUP_EPOCHS" \\
  --blr "$BLR" \\
  --weight_decay "$WEIGHT_DECAY" \\
  --data_path "$DATA_INDEX_ROOT" \\
  --output_dir "$OUTPUT_DIR" \\
  --log_dir "$LOG_DIR" \\
  --experiment "$EXPERIMENT" \\
  --learning_loss \\
  --relative \\
  --norm_pix_loss \\
  --num_workers "$NUM_WORKERS" \\
  "\${EXTRA_ARGS[@]}"

date
echo DONE_HPM_PRETRAIN_${EPOCHS}
EOF

JOB_ID="$(qsub "$JOB_DIR/run_${EXPERIMENT}.pbs")"
echo "submitted=$JOB_ID"
echo "node=$PBS_NODE"
echo "epochs=$EPOCHS"
echo "data=$DATA_DIR"
echo "output=$OUTPUT_DIR"
echo "init_ckpt=$INIT_CKPT"
echo "log=$JOB_DIR/job_${EXPERIMENT}.log"
