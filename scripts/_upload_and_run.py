"""Upload code and run golden-checkpoint experiments on UIC GPU 0."""
import os, paramiko, time, io

HOST = os.environ.get("UIC_HOST", "10.250.93.98")
PORT = int(os.environ.get("UIC_PORT", "6422"))
USER = os.environ.get("UIC_USER", "uic2")
PASSWORD = os.environ["UIC_PASSWORD"]

CHECKPOINTS = [
    ("v3_1_ep201", "/home/uic2/zhaoyi/medical-i-jepa/golden_checkpoints/v3.1_ep201_jepa.pth.tar"),
    ("v4_ep50",    "/home/uic2/zhaoyi/medical-i-jepa/golden_checkpoints/v4_ep50_jepa.pth.tar"),
    ("v5_ep60",    "/home/uic2/zhaoyi/medical-i-jepa/golden_checkpoints/v5_ep60_jepa.pth.tar"),
    ("v6_ep40",    "/home/uic2/zhaoyi/medical-i-jepa/golden_checkpoints/v6_ep40_jepa.pth.tar"),
]

DATA_DIR = "/home/uic2/zhaoyi/VinBigData_ChestXray"
BASE = "/home/uic2/Raymond"
RESULTS = f"{BASE}/results"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)

def ssh(cmd, timeout=120):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode()
    err = stderr.read().decode()
    rc = stdout.channel.recv_exit_status()
    return rc, out, err

# ----- Upload code -----
print("=" * 60)
print("STEP 1: Upload code")
print("=" * 60)
sftp = client.open_sftp()
sftp.put("d:/JEPA/jepa_code_upload.tar.gz", f"{BASE}/jepa_code.tar.gz")
sftp.close()
print("Upload OK")

rc, out, err = ssh(
    f"cd {BASE} && rm -rf experiments scripts tests && "
    f"tar -xzf jepa_code.tar.gz && "
    f"ls experiments/exp7_probe_capacity/run_exp7.py experiments/exp8_partial_finetune/run_exp8.py"
)
print("Extract:", out.strip())
if err:
    print("ERR:", err[:300])

# ----- Build script -----
print("\n" + "=" * 60)
print("STEP 2: Create run script")
print("=" * 60)

lines = [
    "#!/usr/bin/env bash",
    "set -euo pipefail",
    f"cd {BASE}",
    "source /home/uic2/miniconda3/etc/profile.d/conda.sh",
    "conda activate medical_ijepa",
    "export CUDA_VISIBLE_DEVICES=0",
    "export OMP_NUM_THREADS=8",
    "",
    f"DATA_DIR={DATA_DIR}",
    f"RESULTS={RESULTS}",
    "mkdir -p $RESULTS",
    "echo \"Starting golden experiments at $(date)\"",
    "",
]

for short, ckpt in CHECKPOINTS:
    exp2b_dir = f"$RESULTS/exp2b_ijepa_vith_{short}_fixedsplit"
    lines += [
        f"echo '===== Exp2b class-aligned: {short} ====='",
        f"python -u experiments/exp2_lesion_sensitivity/run_exp2_class_aligned.py "
        f"  --model ijepa --weights {ckpt} --data_dir $DATA_DIR "
        f"  --output_dir {exp2b_dir} "
        f"  --num_controls 5 --control_strategy matched_random "
        f"  --max_samples 999999 --batch_size 8 --encoder_batch_size 8 "
        f"  2>&1 | tee {exp2b_dir}/run.log",
        f"echo 'Exp2b {short} done at' $(date)",
        "",
    ]

for short, ckpt in CHECKPOINTS:
    exp2b_dir = f"$RESULTS/exp2b_ijepa_vith_{short}_fixedsplit"
    exp3_dir = f"$RESULTS/exp3_frequency_ijepa_vith_{short}_fixedsplit"
    lines += [
        f"echo '===== Exp3 frequency: {short} ====='",
        f"python -u experiments/exp3_frequency_sensitivity/run_exp3.py "
        f"  --model ijepa --weights {ckpt} --data_dir $DATA_DIR "
        f"  --probe_path {exp2b_dir}/probe.pth "
        f"  --output_dir {exp3_dir} "
        f"  2>&1 | tee {exp3_dir}/run.log",
        f"echo 'Exp3 {short} done at' $(date)",
        "",
    ]

for short, ckpt in CHECKPOINTS:
    if short == "v3_1_ep201":
        continue  # already have for ep201
    exp7_dir = f"$RESULTS/exp7_mlp_probe_ijepa_vith_{short}_fixedsplit"
    lines += [
        f"echo '===== Exp7 MLP probe: {short} ====='",
        f"python -u experiments/exp7_probe_capacity/run_exp7.py "
        f"  --model ijepa --weights {ckpt} --data_dir $DATA_DIR "
        f"  --output_dir {exp7_dir} "
        f"  2>&1 | tee {exp7_dir}/run.log",
        f"echo 'Exp7 {short} done at' $(date)",
        "",
    ]

for short, ckpt in CHECKPOINTS:
    if short == "v3_1_ep201":
        continue
    exp8_dir = f"$RESULTS/exp8_partial_ft_ijepa_vith_{short}_fixedsplit"
    lines += [
        f"echo '===== Exp8 partial FT: {short} ====='",
        f"python -u experiments/exp8_partial_finetune/run_exp8.py "
        f"  --model ijepa --weights {ckpt} --data_dir $DATA_DIR "
        f"  --output_dir {exp8_dir} "
        f"  2>&1 | tee {exp8_dir}/run.log",
        f"echo 'Exp8 {short} done at' $(date)",
        "",
    ]

lines.append("echo '===== ALL DONE at' $(date) ' ====='")
script = "\n".join(lines)

sftp = client.open_sftp()
sftp.putfo(io.BytesIO(script.encode()), f"{BASE}/run_golden_experiments.sh")
sftp.close()
rc, out, err = ssh(f"chmod +x {BASE}/run_golden_experiments.sh")
print("Script uploaded. Preview:")
print(script[:2000])

# ----- Launch -----
print("\n" + "=" * 60)
print("STEP 3: Launch experiments")
print("=" * 60)
rc, out, err = ssh(
    f"cd {BASE} && nohup bash run_golden_experiments.sh > run_golden_nohup.log 2>&1 & echo PID=$!"
)
print("Launch:", out.strip())

time.sleep(2)
rc, out, err = ssh("ps aux | grep -E 'run_golden|run_exp' | grep -v grep")
print("Processes:", out[:500])

rc, out, err = ssh("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader")
print("GPU mem:", out.strip())

print(f"\nMonitor: tail -f {BASE}/run_golden_nohup.log")
client.close()
