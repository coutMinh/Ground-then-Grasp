"""Generate comparable Seen/Unseen/H tables from explicitly selected runs."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from g2g.utils.io import sha256_file, write_json


def harmonic_mean(seen: float, unseen: float) -> float:
    return 2 * seen * unseen / (seen + unseen) if seen + unseen else 0.0


def make_tables(results_root: str, run_ids: list[str], output: str) -> dict:
    import json

    root, destination = Path(results_root), Path(output)
    if not run_ids or len(run_ids) != len(set(run_ids)):
        raise ValueError("Select a nonempty list of distinct benchmark evaluation run IDs.")
    with (root / "registry.csv").open(encoding="utf-8", newline="") as stream:
        registry = {record["run_id"]: record for record in csv.DictReader(stream)}
    selected = {}
    hashes = defaultdict(set)
    for run_id in run_ids:
        record = registry[run_id]
        if not record["kind"].startswith("benchmark_eval_"):
            raise ValueError(
                "Paper tables accept benchmark evaluation runs only, never debug runs."
            )
        path = root / "runs" / run_id / "metrics.json"
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if metrics["status"] != "completed" or metrics["split"] not in {"test_seen", "test_unseen"}:
            raise ValueError("Select completed Seen/Unseen test evaluations.")
        key = (metrics["system"], record["seed"], metrics["split"])
        if key in selected:
            raise ValueError("Duplicate system/seed/split; select one run explicitly.")
        selected[key] = metrics["success_rate"]
        hashes[metrics["split"]].add(record["manifest_sha256"])
    if any(len(values) != 1 for values in hashes.values()):
        raise ValueError("Compared systems must use the same test manifest for each split.")
    grouped = defaultdict(list)
    for system, seed, split in selected:
        if split != "test_seen":
            continue
        seen = selected[(system, seed, "test_seen")]
        unseen_key = (system, seed, "test_unseen")
        if unseen_key not in selected:
            raise ValueError("Every system/seed needs paired Seen and Unseen runs.")
        unseen = selected[unseen_key]
        grouped[system].append((seen, unseen, harmonic_mean(seen, unseen)))
    if sum(len(values) for values in grouped.values()) * 2 != len(selected):
        raise ValueError("Unpaired Unseen evaluation selected.")
    rows = []
    for system, values in sorted(grouped.items()):
        mean = [sum(value[index] for value in values) / len(values) for index in range(3)]
        standard = [
            math.sqrt(
                sum((value[index] - mean[index]) ** 2 for value in values) / (len(values) - 1)
            )
            if len(values) > 1
            else None
            for index in range(3)
        ]
        rows.append(
            {
                "system": system,
                "seeds": len(values),
                **dict(zip(("seen", "unseen", "harmonic"), mean, strict=True)),
                **dict(zip(("seen_std", "unseen_std", "harmonic_std"), standard, strict=True)),
            }
        )
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "main.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [r"\begin{tabular}{lccc}", r"\toprule", r"System & Seen & Unseen & H \\", r"\midrule"]
    for row in rows:
        values = []
        for column in ("seen", "unseen", "harmonic"):
            value = f"{row[column]:.3f}"
            if row[f"{column}_std"] is not None:
                value += r" $\pm$ " + f"{row[f'{column}_std']:.3f}"
            values.append(value)
        lines.append(row["system"].replace("_", r"\_") + " & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (destination / "main.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = {
        "run_ids": run_ids,
        "test_manifest_hashes": {key: list(value)[0] for key, value in hashes.items()},
        "registry_sha256": sha256_file(root / "registry.csv"),
        "rows": rows,
    }
    write_json(destination / "provenance.json", report)
    return report
