"""Environment probe. Never collects environment variable values or secrets."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import psutil

from g2g.utils.io import write_json


def probe(output: str | Path) -> dict:
    import torch

    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(output.parent)
    packages = {}
    for name in (
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "transformers",
        "huggingface-hub",
        "tokenizers",
        "safetensors",
        "librosa",
        "soxr",
        "marimo",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu.append(
                {
                    "name": properties.name,
                    "vram_bytes": properties.total_memory,
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    cuda_smoke = {"passed": False, "reason": "CUDA unavailable"}
    if gpu:
        try:
            matrix = torch.ones(16, 16, device="cuda")
            (matrix @ matrix).sum().item()
            torch.cuda.synchronize()
            cuda_smoke = {"passed": True, "reason": None}
        except RuntimeError as error:
            cuda_smoke = {"passed": False, "reason": str(error)}
    driver = None
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        driver = result.stdout.strip() if result.returncode == 0 else "nvidia-smi failed"
    report = {
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": psutil.virtual_memory().total,
        "memory_available_bytes": psutil.virtual_memory().available,
        "packages": packages,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_smoke": cuda_smoke,
        "gpu": gpu,
        "driver": driver,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "running_as_root": os.geteuid() == 0 if hasattr(os, "geteuid") else None,
        "apt_available": shutil.which("apt-get") is not None,
        "hf_token_configured": bool(os.environ.get("HF_TOKEN")),
        "persistence_across_sessions": "unknown; user must verify",
        "session_time_limit": "unknown; user must verify",
        "background_survives_closed_tab": "unknown; user must verify",
    }
    write_json(output, report)
    return report
