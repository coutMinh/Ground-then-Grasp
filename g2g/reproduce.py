"""Command sequence behind the reported results (split `fullpool-8144`, one seed).

`notebooks/reproduce_molab.py` drives these steps on a GPU runtime. Every step is
a plain `python -m g2g ...` call, so it can also be run from a terminal.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

HF_REPO = "coutMinh/ground-then-grasp"
HF_FILES = {
    "prior": "prior/best.pt",
    "grounder": "grounder/best.pt",
    "compose": "composition/best_compose.json",
}
LGD_REPO = "https://github.com/Fsoft-AIC/LGD.git"
LGD_COMMIT = "501904dc5c30c8fad58d90f7e4f49b4095f11eb0"
MANIFEST_SET = "fullpool-8144"
TRAIN_IMAGES, VAL_IMAGES = 8144, 200
SPLITS = {"test_seen": "seen", "test_unseen": "unseen"}

# (run name, --system, uses fine-tuned grounder, uses frozen composition, uses rerank weight)
SYSTEMS = [
    ("ours", "ours", True, True, True),
    ("prior", "prior", False, False, False),
    ("zs-hard", "zero_shot_hard", False, False, False),
    ("ft-hard", "ft_hard", True, False, False),
    ("mask-peak", "mask_peak", True, False, False),
    ("soft", "soft", True, True, False),
    ("oracle-mask", "oracle_mask", False, False, True),
]


@dataclass(frozen=True)
class Workspace:
    """Runtime layout; keep it outside the Git checkout."""

    root: Path

    @property
    def checkout(self) -> Path:
        return self.root / "repos/ground-then-grasp"

    @property
    def reference(self) -> Path:
        return self.root / "repos/LGD"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def data_root(self) -> Path:
        return self.artifacts / "ga_subset"

    @property
    def cache(self) -> Path:
        return self.artifacts / "inspection/http-cache"

    @property
    def manifests(self) -> Path:
        return self.artifacts / "manifests" / MANIFEST_SET

    @property
    def results(self) -> Path:
        return self.artifacts / "results"

    @property
    def checkpoints(self) -> Path:
        return self.artifacts / "checkpoints"

    @property
    def tables(self) -> Path:
        return self.artifacts / "tables/final"

    @property
    def gallery(self) -> Path:
        return self.artifacts / "gallery"

    @property
    def logs(self) -> Path:
        return self.artifacts / "logs"


def _strings(arguments: list) -> list[str]:
    return [str(argument) for argument in arguments]


def data_commands(ws: Workspace, workers: int = 8) -> list[tuple[str, list[str]]]:
    """Build the manifests, extract only their members, then decode every row."""
    return [
        (
            "manifests",
            _strings(
                ["build-manifests", "--upstream", ws.reference, "--output", ws.manifests]
                + ["--cache-dir", ws.cache, "--train-images", TRAIN_IMAGES]
                + ["--val-images", VAL_IMAGES, "--test-images", 0]
            ),
        ),
        (
            "download",
            _strings(
                ["download-subset", "--manifest", ws.manifests / "all.csv"]
                + ["--data-root", ws.data_root, "--cache-dir", ws.cache, "--workers", workers]
            ),
        ),
        (
            "validate",
            _strings(
                ["validate-data", "--manifest", ws.manifests / "all.csv"]
                + ["--data-root", ws.data_root, "--output", ws.manifests / "validation.json"]
            ),
        ),
    ]


def train_command(ws: Workspace, config: str, run_id: str) -> list[str]:
    return _strings(
        ["train", "--config", ws.checkout / "configs" / config]
        + ["--manifest", ws.manifests / "train.csv", "--val-manifest", ws.manifests / "val.csv"]
        + ["--data-root", ws.data_root, "--results-root", ws.results]
        + ["--run-id", run_id, "--purpose", "benchmark"]
    )


TRAIN_RUNS = {
    "prior": ("prior.yaml", "prior-fullpool-01"),
    "grounder": ("grounder.yaml", "grounder-fullpool-01"),
}
SWEEP_RUN = "compose-fullpool-01"


def sweep_command(ws: Workspace, prior: Path, grounder: Path) -> list[str]:
    """Grid over alpha, temperature and rerank weight on val only."""
    return _strings(
        ["sweep", "--manifest", ws.manifests / "val.csv", "--data-root", ws.data_root]
        + ["--prior-checkpoint", prior, "--grounder-checkpoint", grounder]
        + ["--results-root", ws.results, "--run-id", SWEEP_RUN, "--purpose", "benchmark"]
    )


def evaluation_commands(
    ws: Workspace, prior: Path, grounder: Path, compose: Path
) -> list[tuple[str, list[str]]]:
    """Seven systems on both test splits, with the composition frozen from the val sweep."""
    chosen = json.loads(Path(compose).read_text(encoding="utf-8"))["composition"]
    commands = []
    for name, system, uses_grounder, uses_compose, uses_rerank in SYSTEMS:
        for split, tag in SPLITS.items():
            run_id = f"{name}-full-{tag}-01"
            arguments = (
                ["evaluate", "--manifest", ws.manifests / f"{split}.csv"]
                + ["--data-root", ws.data_root, "--prior-checkpoint", prior]
                + ["--results-root", ws.results, "--run-id", run_id, "--purpose", "benchmark"]
                + ["--system", system, "--alpha", chosen["alpha"]]
                + ["--temperature", chosen["temperature"] if uses_compose else 1.0]
                + ["--rerank-weight", chosen["rerank_weight"] if uses_rerank else 0.0]
            )
            if uses_grounder:
                arguments += ["--grounder-checkpoint", grounder]
            if uses_compose:
                arguments += ["--compose-config", compose]
            commands.append((run_id, _strings(arguments)))
    return commands


def table_command(ws: Workspace, run_ids: list[str]) -> list[str]:
    return _strings(
        ["make-tables", "--results-root", ws.results, "--run-ids", *run_ids]
        + ["--output", ws.tables]
    )


def gallery_command(ws: Workspace, run_id: str, limit: int = 6, seed: int = 42) -> list[str]:
    return _strings(
        ["visualize-predictions", "--run-dir", ws.results / "runs" / run_id]
        + ["--data-root", ws.data_root, "--output", ws.gallery / run_id]
        + ["--limit", limit, "--seed", seed, "--selection", "mixed"]
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_manifests(built: Path, protocol: Path) -> dict[str, bool]:
    """Compare rebuilt split CSVs with the hashes committed in `protocol.json`."""
    expected = json.loads(Path(protocol).read_text(encoding="utf-8"))["splits"]
    return {
        split: sha256(Path(built) / f"{split}.csv") == value["sha256"]
        for split, value in expected.items()
    }


def download_checkpoints(ws: Workspace, token: str | None = None) -> dict[str, Path]:
    """Fetch the released checkpoints and verify them against the repo's SHA256.json."""
    from huggingface_hub import hf_hub_download

    def fetch(name: str) -> Path:
        return Path(hf_hub_download(HF_REPO, name, local_dir=ws.checkpoints, token=token))

    expected = json.loads(fetch("SHA256.json").read_text(encoding="utf-8"))
    paths = {key: fetch(name) for key, name in HF_FILES.items()}
    for key in ("prior", "grounder"):
        if sha256(paths[key]) != expected[HF_FILES[key]]:
            raise ValueError(f"SHA-256 mismatch for {HF_FILES[key]}.")
    return paths
