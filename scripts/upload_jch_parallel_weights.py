"""Parallel chunk uploader for the large I-JEPA checkpoints on the jch server."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import math
import os
import posixpath
import time
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = [
    ROOT / "pretrained" / "ijepa" / "jepa-ep95.pth.tar",
    ROOT / "pretrained" / "ijepa" / "jepa-ep201.pth.tar",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("JCH_HOST", "10.169.143.54"))
    parser.add_argument("--user", default=os.environ.get("JCH_USER", "jchwang"))
    parser.add_argument("--password", default=os.environ.get("JCH_PASSWORD"))
    parser.add_argument("--remote-dir", default="/home/jchwang/ray/pretrained/ijepa")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mib", type=int, default=256)
    parser.add_argument("--io-mib", type=int, default=4)
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
        timeout=25,
        look_for_keys=False,
        allow_agent=False,
    )
    return ssh


def run(ssh: paramiko.SSHClient, cmd: str, timeout: int = 300) -> str:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    rc = stdout.channel.recv_exit_status()
    if rc != 0:
        raise RuntimeError(f"remote command failed rc={rc}: {cmd}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return out


def remote_size(sftp: paramiko.SFTPClient, path: str) -> int | None:
    try:
        return int(sftp.stat(path).st_size)
    except FileNotFoundError:
        return None


def quote(path: str) -> str:
    return "'" + path.replace("'", "'\\''") + "'"


def upload_chunk(
    args: argparse.Namespace,
    local: Path,
    remote_chunk: str,
    offset: int,
    size: int,
    io_size: int,
) -> int:
    ssh = connect(args)
    try:
        sftp = ssh.open_sftp()
        try:
            existing = remote_size(sftp, remote_chunk)
            if existing == size:
                return 0
            if existing is not None:
                sftp.remove(remote_chunk)
            with local.open("rb") as src, sftp.open(remote_chunk, "wb") as dst:
                dst.set_pipelined(True)
                src.seek(offset)
                remaining = size
                while remaining > 0:
                    data = src.read(min(io_size, remaining))
                    if not data:
                        raise IOError(f"unexpected EOF reading {local}")
                    dst.write(data)
                    remaining -= len(data)
            final_size = remote_size(sftp, remote_chunk)
            if final_size != size:
                raise IOError(f"remote chunk size mismatch {remote_chunk}: {final_size} != {size}")
            return size
        finally:
            sftp.close()
    finally:
        ssh.close()


def upload_weight(args: argparse.Namespace, local: Path) -> None:
    total = local.stat().st_size
    remote = posixpath.join(args.remote_dir, local.name)
    chunk_dir = posixpath.join(args.remote_dir, f".{local.name}.chunks")
    chunk_size = args.chunk_mib * 1024 * 1024
    io_size = args.io_mib * 1024 * 1024
    n_chunks = math.ceil(total / chunk_size)

    ssh = connect(args)
    try:
        sftp = ssh.open_sftp()
        try:
            existing = remote_size(sftp, remote)
            if existing == total:
                print(f"SKIP {local.name}: remote file already complete ({total} bytes)", flush=True)
                return
        finally:
            sftp.close()
        run(ssh, f"mkdir -p {quote(chunk_dir)}")
    finally:
        ssh.close()

    tasks = []
    for idx in range(n_chunks):
        offset = idx * chunk_size
        size = min(chunk_size, total - offset)
        remote_chunk = posixpath.join(chunk_dir, f"chunk-{idx:05d}")
        tasks.append((idx, remote_chunk, offset, size))

    print(
        f"UPLOAD {local.name}: {total} bytes as {n_chunks} chunks, "
        f"workers={args.workers}, chunk={args.chunk_mib} MiB",
        flush=True,
    )
    start = time.time()
    uploaded = 0
    done = 0
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(upload_chunk, args, local, remote_chunk, offset, size, io_size): (idx, size)
            for idx, remote_chunk, offset, size in tasks
        }
        for fut in futures.as_completed(future_map):
            idx, size = future_map[fut]
            uploaded += fut.result()
            done += 1
            elapsed = max(time.time() - start, 1e-6)
            effective = (done * chunk_size if done < n_chunks else total)
            rate = effective / elapsed
            print(
                f"PROGRESS {local.name}: chunks {done}/{n_chunks}, "
                f"new_upload={uploaded/1024/1024:.1f} MiB, "
                f"effective={rate/1024/1024:.2f} MiB/s",
                flush=True,
            )

    ssh = connect(args)
    try:
        chunk_paths = " ".join(quote(posixpath.join(chunk_dir, f"chunk-{idx:05d}")) for idx in range(n_chunks))
        tmp = remote + ".assembled"
        bad = remote + f".bad.{int(time.time())}"
        cmd = (
            f"set -euo pipefail; "
            f"cat {chunk_paths} > {quote(tmp)}; "
            f"test $(stat -c%s {quote(tmp)}) -eq {total}; "
            f"if [ -f {quote(remote)} ]; then mv -f {quote(remote)} {quote(bad)}; fi; "
            f"mv -f {quote(tmp)} {quote(remote)}; "
            f"rm -rf {quote(chunk_dir)} {quote(remote + '.part')}; "
            f"ls -lh {quote(remote)}"
        )
        out = run(ssh, cmd, timeout=3600)
        print(out.strip(), flush=True)
    finally:
        ssh.close()


def main() -> None:
    args = parse_args()
    for local in WEIGHTS:
        if not local.exists():
            raise FileNotFoundError(local)
        upload_weight(args, local)
    print("DONE parallel weight upload", flush=True)


if __name__ == "__main__":
    main()
