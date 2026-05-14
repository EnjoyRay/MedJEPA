"""Download selected I-JEPA artifacts from the UIC server.

The script is resumable: files are first written as *.part and renamed after
their size matches the remote file.  Credentials are read from environment
variables so passwords are not stored in the repository.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path

import paramiko


HOST = os.environ.get("UIC_HOST", "10.250.93.98")
PORT = int(os.environ.get("UIC_PORT", "6422"))
USER = os.environ.get("UIC_USER", "uic2")
PASSWORD = os.environ["UIC_PASSWORD"]

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "pretrained" / "ijepa"
LOG_PATH = OUT_DIR / "download_uic_artifacts.log"
WORKERS = int(os.environ.get("UIC_DOWNLOAD_WORKERS", "8"))
PARALLEL_THRESHOLD = 512 * 1024 * 1024
CHUNK_SIZE = 256 * 1024 * 1024
BLOCK_SIZE = 1024 * 1024

FILES = [
    (
        "/home/uic2/zhaoyi/medical-i-jepa/logs/pretrain_mimic_cxr_vith14/jepa-ep95.pth.tar",
        "jepa-ep95.pth.tar",
    ),
    (
        "/home/uic2/zhaoyi/medical-i-jepa/logs/pretrain_mimic_cxr_vith14/jepa-ep201.pth.tar",
        "jepa-ep201.pth.tar",
    ),
    (
        "/home/uic2/Raymond/medical-i-jepa_source_for_jch.tar.gz",
        "medical-i-jepa_source_for_jch.tar.gz",
    ),
]


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def connect_sftp() -> tuple[paramiko.SSHClient, paramiko.SFTPClient]:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        HOST,
        port=PORT,
        username=USER,
        password=PASSWORD,
        timeout=20,
        banner_timeout=30,
        auth_timeout=30,
    )
    return client, client.open_sftp()


def load_completed(state_path: Path) -> set[int]:
    if not state_path.exists():
        return set()
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return {int(i) for i in state.get("completed_chunks", [])}
    except Exception:
        return set()


def save_completed(state_path: Path, completed: set[int]) -> None:
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(
        json.dumps({"completed_chunks": sorted(completed)}, indent=2),
        encoding="utf-8",
    )
    tmp.replace(state_path)


def parallel_download_file(remote: str, local: Path, remote_size: int) -> None:
    part = local.with_name(local.name + ".part")
    state_path = local.with_name(local.name + ".chunks.json")

    # A sequential partial file without chunk state cannot be trusted for random
    # chunk completion, so restart it once when switching to parallel mode.
    if part.exists() and not state_path.exists():
        log(f"Removing non-chunked partial before parallel download: {part.name}")
        part.unlink()

    with part.open("ab") as f:
        f.truncate(remote_size)

    chunks = [
        (idx, start, min(start + CHUNK_SIZE, remote_size))
        for idx, start in enumerate(range(0, remote_size, CHUNK_SIZE))
    ]
    completed = load_completed(state_path)
    completed_bytes = sum(end - start for idx, start, end in chunks if idx in completed)
    work: queue.Queue[tuple[int, int, int] | None] = queue.Queue()
    for item in chunks:
        if item[0] not in completed:
            work.put(item)
    for _ in range(WORKERS):
        work.put(None)

    lock = threading.Lock()
    progress = {"bytes": completed_bytes}
    errors: list[BaseException] = []
    start_time = time.time()
    last_log = {"t": start_time}

    log(
        f"PARALLEL START {local.name}: workers={WORKERS}, "
        f"chunks={len(chunks)}, completed={len(completed)}"
    )

    def worker(worker_id: int) -> None:
        client = None
        sftp = None
        try:
            client, sftp = connect_sftp()
            while True:
                item = work.get()
                if item is None:
                    return
                idx, start, end = item
                pos = start
                with sftp.open(remote, "rb") as rf, part.open("r+b", buffering=0) as lf:
                    rf.seek(start)
                    while pos < end:
                        data = rf.read(min(BLOCK_SIZE, end - pos))
                        if not data:
                            raise IOError(f"Unexpected EOF in chunk {idx} at {pos}")
                        lf.seek(pos)
                        lf.write(data)
                        pos += len(data)
                        now = time.time()
                        with lock:
                            progress["bytes"] += len(data)
                            if now - last_log["t"] >= 30 or progress["bytes"] == remote_size:
                                elapsed = max(now - start_time, 1e-6)
                                rate = progress["bytes"] / elapsed
                                remaining = max(remote_size - progress["bytes"], 0)
                                eta = remaining / rate if rate > 0 else float("inf")
                                log(
                                    f"PROGRESS {local.name} {progress['bytes']}/{remote_size} "
                                    f"({progress['bytes'] / remote_size * 100:.2f}%) "
                                    f"rate={rate / 1024 / 1024:.2f} MiB/s eta={eta / 60:.1f} min"
                                )
                                last_log["t"] = now
                with lock:
                    completed.add(idx)
                    save_completed(state_path, completed)
        except BaseException as exc:
            with lock:
                errors.append(exc)
            log(f"ERROR worker={worker_id}: {exc!r}")
        finally:
            try:
                if sftp is not None:
                    sftp.close()
                if client is not None:
                    client.close()
            except Exception:
                pass

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(WORKERS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        raise RuntimeError(f"Parallel download failed with {len(errors)} errors; first={errors[0]!r}")
    if len(completed) != len(chunks):
        raise RuntimeError(f"Parallel download incomplete: {len(completed)}/{len(chunks)} chunks")


def download_file(sftp: paramiko.SFTPClient, remote: str, local: Path) -> dict:
    remote_size = sftp.stat(remote).st_size
    part = local.with_name(local.name + ".part")

    if local.exists() and local.stat().st_size == remote_size:
        log(f"SKIP complete {local.name} ({remote_size} bytes)")
        return {
            "remote": remote,
            "local": str(local),
            "bytes": remote_size,
            "status": "already_complete",
        }

    existing = part.stat().st_size if part.exists() else 0
    if existing > remote_size:
        log(f"Partial file larger than remote, restarting {local.name}")
        part.unlink()
        existing = 0

    log(f"START {remote} -> {local} remote_size={remote_size} resume_at={existing}")
    start_time = time.time()
    last_log = start_time

    if remote_size >= PARALLEL_THRESHOLD:
        parallel_download_file(remote, local, remote_size)
    else:
        # Fast path: Paramiko's get() uses SFTP prefetching, which is much faster
        # over high-latency VPN links than one blocking read request at a time.
        if existing == 0:
            def callback(transferred: int, total: int) -> None:
                nonlocal last_log
                now = time.time()
                if now - last_log >= 30 or transferred == total:
                    elapsed = max(now - start_time, 1e-6)
                    rate = transferred / elapsed
                    remaining = max(total - transferred, 0)
                    eta = remaining / rate if rate > 0 else float("inf")
                    log(
                        f"PROGRESS {local.name} {transferred}/{total} "
                        f"({transferred / total * 100:.2f}%) "
                        f"rate={rate / 1024 / 1024:.2f} MiB/s eta={eta / 60:.1f} min"
                    )
                    last_log = now

            if part.exists():
                part.unlink()
            sftp.get(
                remote,
                str(part),
                callback=callback,
                prefetch=True,
                max_concurrent_prefetch_requests=64,
            )
        else:
            with sftp.open(remote, "rb") as rf, part.open("ab") as lf:
                rf.seek(existing)
                rf.prefetch(remote_size, max_concurrent_requests=64)
                downloaded = existing
                while downloaded < remote_size:
                    chunk = rf.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    lf.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_log >= 30 or downloaded == remote_size:
                        elapsed = max(now - start_time, 1e-6)
                        rate = max((downloaded - existing) / elapsed, 0.0)
                        remaining = max(remote_size - downloaded, 0)
                        eta = remaining / rate if rate > 0 else float("inf")
                        log(
                            f"PROGRESS {local.name} {downloaded}/{remote_size} "
                            f"({downloaded / remote_size * 100:.2f}%) "
                            f"rate={rate / 1024 / 1024:.2f} MiB/s eta={eta / 60:.1f} min"
                        )
                        last_log = now

    final_size = part.stat().st_size
    if final_size != remote_size:
        raise RuntimeError(f"Incomplete download for {local.name}: {final_size} != {remote_size}")

    if local.exists():
        local.unlink()
    part.rename(local)
    log(f"DONE {local.name} ({remote_size} bytes)")
    return {
        "remote": remote,
        "local": str(local),
        "bytes": remote_size,
        "status": "downloaded",
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    log(f"Connecting to {USER}@{HOST}:{PORT}")

    client, sftp = connect_sftp()
    manifest = []
    try:
        for remote, name in FILES:
            manifest.append(download_file(sftp, remote, OUT_DIR / name))
    finally:
        sftp.close()
        client.close()

    manifest_path = OUT_DIR / "uic_artifacts_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"Wrote manifest {manifest_path}")


if __name__ == "__main__":
    main()
