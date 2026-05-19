"""Inventory UIC checkpoints and GPUs for the JEPA300 fair-comparison rerun.

Credentials are read from environment variables.  The script does not store
passwords and only writes a local JSON manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import paramiko


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results" / "uic_jepa300_checkpoint_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find JEPA300/250 artifacts on UIC.")
    parser.add_argument("--host", default=os.environ.get("UIC_HOST", "10.250.93.98"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("UIC_PORT", "6422")))
    parser.add_argument("--user", default=os.environ.get("UIC_USER", "uic2"))
    parser.add_argument("--base", default=os.environ.get("UIC_JEPA_BASE", "/home/uic2/zhaoyi/medical-i-jepa"))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--skip-torch-meta", action="store_true", help="Only stat paths; do not torch.load metadata.")
    parser.add_argument(
        "--torch-meta-path",
        default="",
        help="Optional exact remote checkpoint path for torch metadata. Avoids loading every 10GB checkpoint.",
    )
    return parser.parse_args()


def connect(args: argparse.Namespace) -> paramiko.SSHClient:
    password = os.environ.get("UIC_PASSWORD")
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
    if password:
        kwargs["password"] = password
    client.connect(**kwargs)
    return client


def ssh(client: paramiko.SSHClient, command: str) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command, get_pty=False)
    stdin.close()
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def remote_inventory_command(base: str, skip_torch_meta: bool, torch_meta_path: str = "") -> str:
    meta_block = "meta = {}"
    if torch_meta_path and not skip_torch_meta:
        meta_block = r"""
meta = {}
if path == TORCH_META_PATH:
  try:
    import torch
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("epoch", "global_step", "arch", "model_name"):
            if key in obj:
                value = obj[key]
                try:
                    json.dumps(value)
                    meta[key] = value
                except TypeError:
                    meta[key] = str(value)
        weights = obj.get("target_encoder") or obj.get("encoder") or obj.get("model") or obj.get("state_dict")
        if isinstance(weights, dict):
            meta["num_weight_keys"] = len(weights)
            first_keys = list(weights.keys())[:8]
            meta["first_weight_keys"] = [str(k) for k in first_keys]
  except Exception as exc:
    meta["metadata_error"] = repr(exc)
"""
    py = f"""
import glob, json, os, subprocess, time
base = {json.dumps(base)}
TORCH_META_PATH = {json.dumps(torch_meta_path)}
patterns = [
    f"{{base}}/logs/**/jepa-ep300.pth.tar",
    f"{{base}}/logs/**/jepa-ep250.pth.tar",
    f"{{base}}/logs/**/jepa-latest.pth.tar",
    f"{{base}}/golden_checkpoints/*300*.pth.tar",
    f"{{base}}/golden_checkpoints/*250*.pth.tar",
    f"{{base}}/golden_checkpoints/*50*.pth.tar",
    f"{{base}}/golden_checkpoints/v*_ep*.pth.tar",
]
paths = []
for pattern in patterns:
    paths.extend(glob.glob(pattern, recursive=True))
seen = set()
checkpoints = []
for path in sorted(paths):
    if path in seen or not os.path.isfile(path):
        continue
    seen.add(path)
    st = os.stat(path)
{textwrap.indent(meta_block.strip(), "    ")}
    checkpoints.append({{
        "path": path,
        "bytes": st.st_size,
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
        "metadata": meta,
    }})
gpu_cmd = "nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits"
try:
    gpu_out = subprocess.check_output(gpu_cmd, shell=True, text=True, stderr=subprocess.STDOUT)
except Exception as exc:
    gpu_out = f"GPU_QUERY_FAILED: {{exc!r}}"
data_checks = []
for candidate in ["/home/uic2/zhaoyi/VinBigData_ChestXray", "/home/uic2/Raymond/VinBigData_ChestXray"]:
    data_checks.append({{"path": candidate, "exists": os.path.isdir(candidate)}})
mae_candidates = []
for pattern in [
    "/home/uic2/Raymond/outputs/mae_huge_mimic_300ep*/checkpoint-299.pth",
    "/home/uic2/zhaoyi/outputs/mae_huge_mimic_300ep*/checkpoint-299.pth",
    "/home/uic2/Raymond/JEPA/outputs/mae_huge_mimic_300ep*/checkpoint-299.pth",
    "/home/uic2/zhaoyi/medical-i-jepa/outputs/mae_huge_mimic_300ep*/checkpoint-299.pth",
]:
    mae_candidates.extend(glob.glob(pattern, recursive=True))
print(json.dumps({{
    "base": base,
    "checkpoints": checkpoints,
    "gpus_csv": gpu_out,
    "data_checks": data_checks,
    "mae300_candidates": sorted(set(mae_candidates)),
}}, indent=2))
"""
    return (
        'PYTHON_BIN=""\n'
        'if [ -f /home/uic2/miniconda3/etc/profile.d/conda.sh ]; then\n'
        '  . /home/uic2/miniconda3/etc/profile.d/conda.sh\n'
        '  conda activate medical_ijepa >/dev/null 2>&1 || true\n'
        '  PYTHON_BIN="$(command -v python || true)"\n'
        "fi\n"
        'if [ -z "$PYTHON_BIN" ]; then PYTHON_BIN="$(command -v python3 || command -v python || true)"; fi\n'
        'if [ -z "$PYTHON_BIN" ]; then echo "No python/python3 found on remote PATH" >&2; exit 127; fi\n'
        '"$PYTHON_BIN" - <<\'PY\'\n'
        + py.strip()
        + "\nPY"
    )


def main() -> int:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        client = connect(args)
    except Exception as exc:
        print(
            "Failed to connect to UIC. Set UIC_PASSWORD or configure SSH key auth. "
            f"Error: {exc!r}",
            file=sys.stderr,
        )
        return 2
    try:
        rc, out, err = ssh(client, remote_inventory_command(args.base, args.skip_torch_meta, args.torch_meta_path))
    finally:
        client.close()
    if rc != 0:
        print(err, file=sys.stderr)
        return rc
    out_path.write_text(out, encoding="utf-8")
    print(out)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
