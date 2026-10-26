"""Check HF metadata and HTTP access without downloading dataset archives."""

from __future__ import annotations

import os

import requests

from g2g.utils.io import write_json

DATASETS = {
    "airvlab/Grasp-Anything": ["image_part_aa", "image_part_ab"],
    "airvlab/Grasp-Anything-pp": [
        "grasp_instructions.zip",
        "part_mask.zip",
        "grasp_label_positive.zip",
    ],
}


def check_access(output: str) -> dict:
    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    report = {"token_configured": bool(token), "owner_terms_confirmed": False, "datasets": {}}
    with requests.Session() as session:
        for repository, filenames in DATASETS.items():
            entry = {"metadata_status": None, "gated": None, "files": {}}
            try:
                response = session.get(
                    f"https://huggingface.co/api/datasets/{repository}",
                    headers=headers,
                    timeout=30,
                )
                entry["metadata_status"] = response.status_code
                if response.ok:
                    metadata = response.json()
                    entry["gated"] = metadata.get("gated")
                    entry["revision"] = metadata.get("sha")
                    available = {item["rfilename"] for item in metadata.get("siblings", [])}
                    for filename in filenames:
                        if filename not in available:
                            entry["files"][filename] = {"status": "not listed in metadata"}
                            continue
                        # HEAD follows redirects but retrieves no archive body. requests strips
                        # Authorization on cross-host redirects. Never log signed URLs/headers.
                        with session.head(
                            f"https://huggingface.co/datasets/{repository}/resolve/main/{filename}",
                            headers=headers,
                            allow_redirects=True,
                            timeout=30,
                        ) as head:
                            entry["files"][filename] = {"http_status": head.status_code}
                response.close()
            except requests.RequestException:
                entry["error"] = "Network request failed; no credential or signed URL logged."
            report["datasets"][repository] = entry
    report["interpretation"] = (
        "HTTP access is technical availability only. Confirm the owner's download form/license "
        "separately before downloading. No archive bodies were downloaded."
    )
    write_json(output, report)
    return report
