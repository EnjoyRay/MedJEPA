"""Upload fixed experiment code and UIC I-JEPA artifacts to the jch A800 server.

The large checkpoint uploads are resumable through ``*.part`` files on the
remote side.  Credentials are read from environment variables so the script does
not persist passwords in the repository.
"""

from __future__ import annotations

import argparse
import os
import posixpath
import sys
import time
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parents[1]

CODE_FILES = [
    "MECHANISM_EXPERIMENTS_RUNBOOK.md",
    "experiments/__init__.py",
    "experiments/exp1_robustness/__init__.py",
    "experiments/exp1_robustness/evaluate_robustness.py",
    "experiments/exp1_robustness/linear_probe.py",
    "experiments/exp1_robustness/run_exp1.py",
    "experiments/exp2_lesion_sensitivity/__init__.py",
    "experiments/exp2_lesion_sensitivity/analyze_exp2.py",
    "experiments/exp2_lesion_sensitivity/analyze_class_aligned.py",
    "experiments/exp2_lesion_sensitivity/evaluate_class_aligned.py",
    "experiments/exp2_lesion_sensitivity/evaluate_sensitivity.py",
    "experiments/exp2_lesion_sensitivity/make_class_aligned_cases.py",
    "experiments/exp2_lesion_sensitivity/make_case_studies.py",
    "experiments/exp2_lesion_sensitivity/occlusion.py",
    "experiments/exp2_lesion_sensitivity/run_exp2_class_aligned.py",
    "experiments/exp2_lesion_sensitivity/run_exp2.py",
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
    "experiments/shared/__init__.py",
    "experiments/shared/metrics.py",
    "experiments/shared/model_wrappers.py",
    "experiments/shared/perturbations.py",
    "experiments/shared/vindr_dataset.py",
    "experiments/visualize_predictions.py",
    "scripts/build_class_aligned_figures.py",
    "scripts/build_paper_figures.py",
    "scripts/run_exp6_nca_jch.sh",
    "scripts/run_mechanism_experiments_a800.sh",
    "scripts/run_mae300_downstream_jch.sh",
    "scripts/run_mae_downstream_a800.sh",
    "scripts/upload_jch_artifacts.py",
    "scripts/write_exp2b_report_section.py",
]

ARTIFACT_FILES = [
    "pretrained/ijepa/medical-i-jepa_source_for_jch.tar.gz",
    "pretrained/ijepa/jepa-ep95.pth.tar",
    "pretrained/ijepa/jepa-ep201.pth.tar",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("JCH_HOST", "10.169.143.54"))
    parser.add_argument("--user", default=os.environ.get("JCH_USER", "jchwang"))
    parser.add_argument("--password", default=os.environ.get("JCH_PASSWORD"))
    parser.add_argument("--remote-root", default="/home/jchwang/ray/JEPA")
    parser.add_argument("--remote-artifacts", default="/home/jchwang/ray/pretrained/ijepa")
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--progress-sec", type=float, default=20.0)
    parser.add_argument("--skip-large", action="store_true")
    return parser.parse_args()


def connect(args: argparse.Namespace) -> paramiko.SSHClient:
    if not args.password:
        raise SystemExit("Set JCH_PASSWORD in the environment.")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        args.host,
        username=args.user,
        password=args.password,
        timeout=20,
        look_for_keys=False,
        allow_agent=False,
    )
    return ssh


def run(ssh: paramiko.SSHClient, cmd: str) -> str:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=120)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    if rc != 0:
        raise RuntimeError(f"remote command failed rc={rc}: {cmd}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return out


def mkdir_p(ssh: paramiko.SSHClient, remote_dir: str) -> None:
    quoted = "'" + remote_dir.replace("'", "'\\''") + "'"
    run(ssh, f"mkdir -p {quoted}")


def remote_size(sftp: paramiko.SFTPClient, path: str) -> int | None:
    try:
        return int(sftp.stat(path).st_size)
    except FileNotFoundError:
        return None


def upload_file(
    ssh: paramiko.SSHClient,
    sftp: paramiko.SFTPClient,
    local: Path,
    remote: str,
    *,
    chunk_size: int,
    progress_sec: float,
) -> None:
    total = local.stat().st_size
    remote_dir = posixpath.dirname(remote)
    mkdir_p(ssh, remote_dir)

    existing = remote_size(sftp, remote)
    if existing == total:
        print(f"SKIP {local} -> {remote} ({total} bytes already present)", flush=True)
        return

    part = remote + ".part"
    part_size = remote_size(sftp, part) or 0
    if part_size > total:
        sftp.remove(part)
        part_size = 0

    mode = "ab" if part_size else "wb"
    sent = part_size
    start = time.time()
    last = start
    print(f"UPLOAD {local} -> {remote} start={sent}/{total}", flush=True)
    with local.open("rb") as src, sftp.open(part, mode) as dst:
        dst.set_pipelined(True)
        src.seek(sent)
        while True:
            data = src.read(chunk_size)
            if not data:
                break
            dst.write(data)
            sent += len(data)
            now = time.time()
            if now - last >= progress_sec:
                rate = (sent - part_size) / max(now - start, 1e-6)
                pct = 100.0 * sent / total
                eta = (total - sent) / max(rate, 1e-6)
                print(
                    f"PROGRESS {local.name}: {sent}/{total} bytes "
                    f"({pct:.2f}%), {rate/1024/1024:.2f} MiB/s, eta {eta/60:.1f} min",
                    flush=True,
                )
                last = now
    final_size = remote_size(sftp, part)
    if final_size != total:
        raise RuntimeError(f"partial upload size mismatch for {remote}: {final_size} != {total}")
    if existing is not None and existing != total:
        bad = f"{remote}.bad.{int(time.time())}"
        sftp.rename(remote, bad)
    sftp.rename(part, remote)
    elapsed = time.time() - start
    rate = (total - part_size) / max(elapsed, 1e-6)
    print(f"DONE {local.name}: {total} bytes, {rate/1024/1024:.2f} MiB/s", flush=True)


def main() -> None:
    args = parse_args()
    chunk_size = args.chunk_mib * 1024 * 1024
    ssh = connect(args)
    try:
        sftp = ssh.open_sftp()
        try:
            for rel in CODE_FILES:
                local = ROOT / rel
                if not local.exists():
                    print(f"WARN missing local file: {local}", file=sys.stderr, flush=True)
                    continue
                remote = posixpath.join(args.remote_root, *Path(rel).parts)
                upload_file(
                    ssh,
                    sftp,
                    local,
                    remote,
                    chunk_size=chunk_size,
                    progress_sec=max(args.progress_sec, 1.0),
                )
            for rel in ARTIFACT_FILES:
                if args.skip_large and rel.endswith(".pth.tar"):
                    continue
                local = ROOT / rel
                if not local.exists():
                    raise FileNotFoundError(local)
                remote = posixpath.join(args.remote_artifacts, Path(rel).name)
                upload_file(
                    ssh,
                    sftp,
                    local,
                    remote,
                    chunk_size=chunk_size,
                    progress_sec=args.progress_sec,
                )
        finally:
            sftp.close()

        run(
            ssh,
            "mkdir -p /home/jchwang/ray/zhaoyi && "
            "tar -xzf /home/jchwang/ray/pretrained/ijepa/medical-i-jepa_source_for_jch.tar.gz "
            "-C /home/jchwang/ray/zhaoyi",
        )
        print("EXTRACTED medical-i-jepa source under /home/jchwang/ray/zhaoyi", flush=True)
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
