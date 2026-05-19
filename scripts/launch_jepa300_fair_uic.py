"""Upload code to UIC and launch the JEPA300 fair-comparison script.

Credentials are read from UIC_PASSWORD or SSH key auth.  Remote execution is
started with nohup so it survives the local terminal.
"""

from __future__ import annotations

import argparse
import os
import posixpath
import shlex
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE = "/home/uic2/Raymond/JEPA"

FILES = [
    "experiments",
    "scripts/export_ijepa_encoder_checkpoint.py",
    "scripts/run_jepa300_fair_uic.sh",
    "scripts/build_jepa300_fair_summary.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch JEPA300 fair comparison on UIC.")
    parser.add_argument("--host", default=os.environ.get("UIC_HOST", "10.250.93.98"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("UIC_PORT", "6422")))
    parser.add_argument("--user", default=os.environ.get("UIC_USER", "uic2"))
    parser.add_argument("--remote", default=os.environ.get("UIC_REMOTE_REPO", DEFAULT_REMOTE))
    parser.add_argument("--mode", default=os.environ.get("MODE", "smoke"), choices=["smoke", "full"])
    parser.add_argument("--job-kind", default=os.environ.get("JOB_KIND", "core"))
    parser.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "/home/uic2/zhaoyi/VinBigData_ChestXray"))
    parser.add_argument("--mae300", default=os.environ.get("MAE300", ""))
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


def connect(args: argparse.Namespace) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kwargs = {
        "hostname": args.host,
        "port": args.port,
        "username": args.user,
        "timeout": 20,
        "banner_timeout": 30,
        "auth_timeout": 30,
    }
    password = os.environ.get("UIC_PASSWORD")
    if password:
        kwargs["password"] = password
    client.connect(**kwargs)
    return client


def ssh(client: paramiko.SSHClient, cmd: str) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(cmd)
    stdin.close()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def ensure_remote_dir(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = remote_dir.strip("/").split("/")
    path = ""
    for part in parts:
        path += "/" + part
        try:
            sftp.stat(path)
        except FileNotFoundError:
            sftp.mkdir(path)


def upload_file(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    ensure_remote_dir(sftp, posixpath.dirname(remote))
    try:
        st = sftp.stat(remote)
        if st.st_size == local.stat().st_size:
            return
    except FileNotFoundError:
        pass
    sftp.put(str(local), remote)
    print(f"UPLOAD {local.relative_to(ROOT)}")


def upload_path(sftp: paramiko.SFTPClient, rel: str, remote_base: str) -> None:
    local = ROOT / rel
    if local.is_file():
        upload_file(sftp, local, posixpath.join(remote_base, rel.replace("\\", "/")))
        return
    for path in local.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts:
            remote = posixpath.join(remote_base, str(path.relative_to(ROOT)).replace("\\", "/"))
            upload_file(sftp, path, remote)


def main() -> int:
    args = parse_args()
    client = connect(args)
    try:
        if not args.no_upload:
            sftp = client.open_sftp()
            try:
                for rel in FILES:
                    upload_path(sftp, rel, args.remote)
            finally:
                sftp.close()
        rc, out, err = ssh(client, f"chmod +x {shlex.quote(args.remote)}/scripts/run_jepa300_fair_uic.sh")
        if rc != 0:
            raise RuntimeError(err)
        log = f"/home/{args.user}/Raymond/jobs/jepa300_fair/nohup_{args.job_kind}_{args.mode}.log"
        env = {
            "MODE": args.mode,
            "JOB_KIND": args.job_kind,
            "DATA_DIR": args.data_dir,
            "CUDA_VISIBLE_DEVICES": args.gpu,
        }
        if args.mae300:
            env["MAE300"] = args.mae300
        prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
        cmd = (
            f"cd {shlex.quote(args.remote)} && mkdir -p /home/{args.user}/Raymond/jobs/jepa300_fair && "
            f"nohup env {prefix} bash scripts/run_jepa300_fair_uic.sh "
            f"> {shlex.quote(log)} 2>&1 < /dev/null & echo PID=$!"
        )
        rc, out, err = ssh(client, cmd)
        if rc != 0:
            raise RuntimeError(err)
        print(out.strip())
        print(f"Monitor: ssh -p {args.port} {args.user}@{args.host} 'tail -f {log}'")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
