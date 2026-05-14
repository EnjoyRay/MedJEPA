"""Download completed Exp2b class-aligned outputs from the jch A800 server."""

from __future__ import annotations

import argparse
import os
import posixpath
import stat
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parents[1]

REMOTE_ITEMS = [
    "/home/jchwang/ray/JEPA/results/exp2b_dryrun_ijepa_vith_ep201",
    "/home/jchwang/ray/JEPA/results/exp2b_class_aligned_ijepa_vith_ep95_fixedsplit",
    "/home/jchwang/ray/JEPA/results/exp2b_class_aligned_ijepa_vith_ep201_fixedsplit",
    "/home/jchwang/ray/JEPA/results/exp2b_class_aligned_mae_huge_mimic_97ep_fixedsplit",
    "/home/jchwang/ray/JEPA/results/exp2b_class_aligned_mae_huge_mimic_300ep_fixedsplit",
    "/home/jchwang/ray/JEPA/results/paper_figures/fig9_exp2b_class_aligned_overall.png",
    "/home/jchwang/ray/JEPA/results/paper_figures/fig10_exp2b_group_effects.png",
    "/home/jchwang/ray/JEPA/results/paper_figures/fig11_exp2b_class_heatmap.png",
    "/home/jchwang/ray/JEPA/results/paper_figures/fig12_exp2b_area_response.png",
    "/home/jchwang/ray/JEPA/results/paper_figures/fig13_exp2b_case_sheet.png",
    "/home/jchwang/ray/JEPA/results/paper_figures/table_exp2b_class_aligned_overall.csv",
    "/home/jchwang/ray/JEPA/results/paper_figures/table_exp2b_class_aligned_groups.csv",
    "/home/jchwang/ray/JEPA/results/paper_figures/table_exp2b_class_aligned_classes.csv",
    "/home/jchwang/ray/JEPA/results/paper_figures/table_exp2b_class_aligned_samples.csv",
    "/home/jchwang/ray/JEPA/results/paper_figures/exp2b_report_section.md",
    "/home/jchwang/ray/jobs/exp2b_class_aligned/job.log",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("JCH_HOST", "10.169.143.54"))
    parser.add_argument("--user", default=os.environ.get("JCH_USER", "jchwang"))
    parser.add_argument("--password", default=os.environ.get("JCH_PASSWORD"))
    parser.add_argument("--remote-root", default="/home/jchwang/ray/JEPA")
    parser.add_argument("--local-root", default=str(ROOT))
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


def is_dir(sftp: paramiko.SFTPClient, remote: str) -> bool:
    try:
        return stat.S_ISDIR(sftp.stat(remote).st_mode)
    except FileNotFoundError:
        return False


def exists(sftp: paramiko.SFTPClient, remote: str) -> bool:
    try:
        sftp.stat(remote)
        return True
    except FileNotFoundError:
        return False


def remote_to_local(remote: str, remote_root: str, local_root: Path) -> Path:
    if remote.startswith(remote_root + "/"):
        rel = remote[len(remote_root) + 1 :]
    elif remote.startswith("/home/jchwang/ray/jobs/"):
        rel = "remote_jobs/" + remote[len("/home/jchwang/ray/jobs/") :]
    else:
        rel = remote.lstrip("/").replace("/", "_")
    return local_root / Path(*rel.split("/"))


def download_file(sftp: paramiko.SFTPClient, remote: str, local: Path) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote, str(local))
    print(f"GET {remote} -> {local}")


def download_dir(sftp: paramiko.SFTPClient, remote: str, local: Path) -> None:
    local.mkdir(parents=True, exist_ok=True)
    for item in sftp.listdir_attr(remote):
        child_remote = posixpath.join(remote, item.filename)
        child_local = local / item.filename
        if stat.S_ISDIR(item.st_mode):
            download_dir(sftp, child_remote, child_local)
        else:
            download_file(sftp, child_remote, child_local)


def main() -> None:
    args = parse_args()
    local_root = Path(args.local_root)
    ssh = connect(args)
    try:
        sftp = ssh.open_sftp()
        try:
            for remote in REMOTE_ITEMS:
                if not exists(sftp, remote):
                    print(f"SKIP missing {remote}")
                    continue
                local = remote_to_local(remote, args.remote_root.rstrip("/"), local_root)
                if is_dir(sftp, remote):
                    download_dir(sftp, remote, local)
                else:
                    download_file(sftp, remote, local)
        finally:
            sftp.close()
    finally:
        ssh.close()


if __name__ == "__main__":
    main()
