"""Upload experiment code to UIC server."""
from __future__ import annotations

import os, sys, time
from pathlib import Path
import paramiko

ROOT = Path(__file__).resolve().parents[1]

FILES_TO_UPLOAD = [
    "experiments/__init__.py",
    "experiments/shared/__init__.py",
    "experiments/shared/metrics.py",
    "experiments/shared/model_wrappers.py",
    "experiments/shared/perturbations.py",
    "experiments/shared/vindr_dataset.py",
    "experiments/exp1_robustness/__init__.py",
    "experiments/exp1_robustness/evaluate_robustness.py",
    "experiments/exp1_robustness/linear_probe.py",
    "experiments/exp1_robustness/run_exp1.py",
    "experiments/exp2_lesion_sensitivity/__init__.py",
    "experiments/exp2_lesion_sensitivity/analyze_class_aligned.py",
    "experiments/exp2_lesion_sensitivity/evaluate_class_aligned.py",
    "experiments/exp2_lesion_sensitivity/occlusion.py",
    "experiments/exp2_lesion_sensitivity/run_exp2_class_aligned.py",
    "experiments/exp2_lesion_sensitivity/make_class_aligned_cases.py",
    "experiments/exp3_frequency_sensitivity/__init__.py",
    "experiments/exp3_frequency_sensitivity/frequency_perturbations.py",
    "experiments/exp3_frequency_sensitivity/plot_exp3.py",
    "experiments/exp3_frequency_sensitivity/run_exp3.py",
    "experiments/exp4_mechanism/__init__.py",
    "experiments/exp4_mechanism/plot_exp4.py",
    "experiments/exp4_mechanism/run_exp4.py",
    "experiments/exp5_lightweight_mitigation/__init__.py",
    "experiments/exp5_lightweight_mitigation/denoise.py",
    "experiments/exp5_lightweight_mitigation/plot_exp5.py",
    "experiments/exp5_lightweight_mitigation/run_exp5.py",
    "experiments/exp6_noise_consistent_adapter/__init__.py",
    "experiments/exp6_noise_consistent_adapter/adapter.py",
    "experiments/exp6_noise_consistent_adapter/plot_exp6.py",
    "experiments/exp6_noise_consistent_adapter/run_exp6.py",
    "experiments/exp7_probe_capacity/__init__.py",
    "experiments/exp7_probe_capacity/plot_probe_capacity.py",
    "experiments/exp7_probe_capacity/run_exp7.py",
    "experiments/exp8_partial_finetune/__init__.py",
    "experiments/exp8_partial_finetune/run_exp8.py",
    "scripts/build_jepa300_fair_summary.py",
    "scripts/export_ijepa_encoder_checkpoint.py",
    "scripts/inventory_uic_jepa300.py",
    "scripts/launch_jepa300_fair_uic.py",
    "scripts/run_jepa300_fair_uic.sh",
    "tests/smoke_test.py",
]

HOST = os.environ.get("UIC_HOST", "10.250.93.98")
PORT = int(os.environ.get("UIC_PORT", "6422"))
USER = os.environ.get("UIC_USER", "uic2")
REMOTE_BASE = "/home/uic2/Raymond"


def main():
    password = os.environ["UIC_PASSWORD"]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=password, timeout=20)
    sftp = client.open_sftp()

    def ensure_dir(remote_dir: str):
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            dirs = remote_dir.strip("/").split("/")
            for i in range(1, len(dirs) + 1):
                partial = "/" + "/".join(dirs[:i])
                try:
                    sftp.stat(partial)
                except FileNotFoundError:
                    sftp.mkdir(partial)

    uploaded = 0
    for rel in FILES_TO_UPLOAD:
        local = ROOT / rel
        if not local.exists():
            print(f"SKIP missing: {local}", flush=True)
            continue
        remote = f"{REMOTE_BASE}/{rel}"
        ensure_dir(str(Path(remote).parent))
        local_size = local.stat().st_size
        try:
            remote_stat = sftp.stat(remote)
            if remote_stat.st_size == local_size:
                continue  # already up to date
        except FileNotFoundError:
            pass
        sftp.put(str(local), remote)
        uploaded += 1
        print(f"UPLOAD {rel}", flush=True)

    sftp.close()
    client.close()
    print(f"Uploaded {uploaded} files.", flush=True)


if __name__ == "__main__":
    main()
