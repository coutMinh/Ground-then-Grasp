# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["marimo==0.25.2", "pip>=24"]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(
    width="medium",
    app_title="Ground-then-Grasp: reproduce on molab",
)


@app.cell
def _():
    import csv
    import json
    import os
    import subprocess
    import sys
    import threading
    import time
    from pathlib import Path

    import marimo as mo

    return Path, csv, json, mo, os, subprocess, sys, threading, time


@app.cell
def _(mo):
    mo.md(r"""
    # Ground-then-Grasp: reproduce the reported results

    Run on a GPU runtime (molab). Each step has its own button and can be re-run;
    finished runs are skipped. Set `HF_TOKEN` as a runtime secret after accepting the
    terms of `airvlab/Grasp-Anything` and `airvlab/Grasp-Anything-pp`.

    | Step | What | Time on one RTX PRO 6000 |
    |---|---|---|
    | 1 | clone, install, GPU probe, unit tests, pinned LGD split lists | ~5 min |
    | 2 | build the `fullpool-8144` manifests, download only their files, validate | 1–1.5 h |
    | 3 | **HF mode:** download released checkpoints · **Train mode:** R1 prior ∥ R2 grounder, then val sweep | 1 min · ~75 min |
    | 4 | evaluate 7 systems on Seen/Unseen, build the table, compare with the committed one | ~12 min |
    | 5 | draw saved predictions against ground truth | ~1 min |
    """)
    return


@app.cell
def _(Path, mo):
    config = mo.ui.dictionary(
        {
            "repo_url": mo.ui.text(
                value="https://github.com/coutMinh/Ground-then-Grasp.git",
                label="Repository",
                full_width=True,
            ),
            "ref": mo.ui.text(value="main", label="Branch / tag / commit"),
            "workspace": mo.ui.text(
                value=str(Path.home() / "g2g-workspace"), label="Workspace (outside Git)"
            ),
            "mode": mo.ui.dropdown(
                options=["Use HF checkpoints", "Train from scratch"],
                value="Use HF checkpoints",
                label="Checkpoints",
            ),
            "workers": mo.ui.number(start=1, stop=32, value=8, label="Download workers"),
        }
    ).form(submit_button_label="Save settings")
    mo.vstack(
        [
            config,
            mo.md(
                "The paper's numbers were produced at commit `9929373`; later commits only "
                "reorganise the repository."
            ),
        ]
    )
    return (config,)


@app.cell
def _(Path, config, mo, os, subprocess, sys, time):
    mo.stop(config.value is None, mo.md("Save the settings to continue."))
    settings = config.value
    root = Path(settings["workspace"]).expanduser().resolve()
    checkout = root / "repos/ground-then-grasp"
    logs = root / "artifacts/logs"
    logs.mkdir(parents=True, exist_ok=True)

    def environment():
        values = os.environ.copy()
        values.setdefault("HF_HOME", str(root / "artifacts/cache/huggingface"))
        values["PIP_CACHE_DIR"] = str(root / "artifacts/cache/pip")
        values["LGD_REFERENCE_ROOT"] = str(root / "repos/LGD")
        return values

    def run(arguments, stage, cwd=None):
        """Run one command; its full output goes to a log file, never to the notebook."""
        log = logs / f"{stage}-{int(time.time() * 1000)}.log"
        with log.open("w", encoding="utf-8") as sink:
            process = subprocess.run(
                [str(argument) for argument in arguments],
                cwd=cwd,
                env=environment(),
                stdout=sink,
                stderr=subprocess.STDOUT,
            )
        if process.returncode:
            tail = "".join(log.read_text(encoding="utf-8", errors="replace").splitlines(True)[-40:])
            raise RuntimeError(f"{stage} failed (exit {process.returncode}), log {log}\n{tail}")
        return log

    def g2g(arguments, stage):
        return run([sys.executable, "-m", "g2g", *arguments], stage, cwd=checkout)

    return checkout, g2g, root, run, settings


@app.cell
def _(mo):
    setup_button = mo.ui.run_button(label="1. Setup")
    setup_button
    return (setup_button,)


@app.cell
def _(
    checkout,
    g2g,
    json,
    mo,
    root,
    run,
    settings,
    setup_button,
    subprocess,
    sys,
):
    mo.stop(not setup_button.value, mo.md("Press **1. Setup**."))
    if not (checkout / ".git").exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--", settings["repo_url"], checkout], "clone")
    run(["git", "-C", checkout, "fetch", "origin", settings["ref"]], "fetch")
    run(["git", "-C", checkout, "checkout", "--detach", "FETCH_HEAD"], "checkout")
    run([sys.executable, "-m", "pip", "install", "-e", f"{checkout}[grounder,dev]"], "install")
    if str(checkout) not in sys.path:
        sys.path.insert(0, str(checkout))
    from g2g import reproduce as R

    ws = R.Workspace(root)
    g2g(["probe", "--output", ws.artifacts / "probe.json"], "probe")
    probe = json.loads((ws.artifacts / "probe.json").read_text(encoding="utf-8"))
    if not probe["cuda_smoke"]["passed"]:
        raise RuntimeError(f"CUDA check failed: {probe['cuda_smoke']['reason']}")
    run([sys.executable, "-m", "pytest", "-q"], "pytest", cwd=checkout)
    if not ws.reference.exists():
        run(["git", "clone", "--", R.LGD_REPO, ws.reference], "lgd-clone")
        run(["git", "-C", ws.reference, "checkout", "--detach", R.LGD_COMMIT], "lgd-pin")
    g2g(
        [
            "audit-upstream",
            "--root",
            ws.reference,
            "--output",
            ws.artifacts / "upstream_splits.json",
        ],
        "lgd-audit",
    )
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    gpu_name = probe["gpu"][0]["name"]
    mo.md(f"Setup done. Commit `{commit[:7]}`, GPU `{gpu_name}`.")
    return R, ws


@app.cell
def _(mo):
    data_button = mo.ui.run_button(label="2. Data")
    data_button
    return (data_button,)


@app.cell
def _(R, data_button, g2g, json, mo, settings, time, ws):
    mo.stop(not data_button.value, mo.md("Press **2. Data**."))
    for _stage, _arguments in R.data_commands(ws, settings["workers"]):
        if _stage == "manifests" and (ws.manifests / "all.csv").is_file():
            continue
        for _attempt in range(8 if _stage == "download" else 1):
            try:
                g2g(_arguments, _stage)
                break
            except RuntimeError:
                # HF CDN reads can time out; the archive cache makes a retry resume.
                if _stage != "download" or _attempt == 7:
                    raise
                time.sleep(20)
    hashes = R.check_manifests(
        ws.manifests, ws.checkout / "data/manifests" / R.MANIFEST_SET / "protocol.json"
    )
    validation = json.loads((ws.manifests / "validation.json").read_text(encoding="utf-8"))
    if not all(hashes.values()) or validation["failure_count"]:
        raise RuntimeError(
            f"Manifest hashes {hashes}; validation failures {validation['failure_count']}"
        )
    data_ready = True
    mo.md(
        f"Manifests match the committed hashes ({', '.join(hashes)}). "
        f"{validation['rows']} rows, {validation['decoded_images']} images, 0 failures."
    )
    return (data_ready,)


@app.cell
def _(mo):
    checkpoint_button = mo.ui.run_button(label="3. Checkpoints")
    checkpoint_button
    return (checkpoint_button,)


@app.cell
def _(
    Path,
    R,
    checkpoint_button,
    data_ready,
    g2g,
    json,
    mo,
    os,
    settings,
    threading,
    time,
    ws,
):
    mo.stop(not (checkpoint_button.value and data_ready), mo.md("Press **3. Checkpoints**."))
    if settings["mode"] == "Use HF checkpoints":
        paths = R.download_checkpoints(ws, os.environ.get("HF_TOKEN"))
    else:
        paths, errors = {}, {}

        def _train(task, config_name, run_id):
            best = ws.results / "runs" / run_id / "best.pt"
            try:
                if not best.is_file():
                    g2g(R.train_command(ws, config_name, run_id), "train-" + run_id)
                paths[task] = best
            except Exception as error:  # reported below, after both threads finish
                errors[task] = error

        threads = []
        for _task, (_config_name, _run_id) in R.TRAIN_RUNS.items():
            threads.append(threading.Thread(target=_train, args=(_task, _config_name, _run_id)))
            threads[-1].start()
            time.sleep(150)  # stagger start-up; simultaneous starts failed once on molab
        for _thread in threads:
            _thread.join()
        if errors:
            raise RuntimeError(errors)
        paths["compose"] = ws.results / "runs" / R.SWEEP_RUN / "best_compose.json"
        if not paths["compose"].is_file():
            g2g(R.sweep_command(ws, paths["prior"], paths["grounder"]), "sweep")
    paths = {key: Path(value) for key, value in paths.items()}
    chosen = json.loads(paths["compose"].read_text(encoding="utf-8"))["composition"]
    mo.md(
        f"Prior `{paths['prior']}`\n\nGrounder `{paths['grounder']}`\n\n"
        f"Composition chosen on val: alpha={chosen['alpha']}, temperature={chosen['temperature']}, "
        f"rerank weight={chosen['rerank_weight']}."
    )
    return (paths,)


@app.cell
def _(mo):
    evaluate_button = mo.ui.run_button(label="4. Evaluate + table")
    evaluate_button
    return (evaluate_button,)


@app.cell
def _(R, csv, evaluate_button, g2g, mo, paths, ws):
    mo.stop(not evaluate_button.value, mo.md("Press **4. Evaluate + table**."))
    run_ids = []
    for _run_id, _arguments in R.evaluation_commands(
        ws, paths["prior"], paths["grounder"], paths["compose"]
    ):
        if not (ws.results / "runs" / _run_id / "metrics.json").is_file():
            g2g(_arguments, "eval-" + _run_id)
        run_ids.append(_run_id)
    g2g(R.table_command(ws, run_ids), "tables")

    def _read(path):
        with path.open(encoding="utf-8") as stream:
            return {row["system"]: row for row in csv.DictReader(stream)}

    reference = _read(ws.checkout / "results" / R.MANIFEST_SET / "tables/main.csv")
    rows = []
    for _system, _row in _read(ws.tables / "main.csv").items():
        entry = {"system": _system}
        for _column in ("seen", "unseen", "harmonic"):
            entry[_column] = round(100 * float(_row[_column]), 2)
            if _system in reference:
                entry[_column + " (paper)"] = round(100 * float(reference[_system][_column]), 2)
        rows.append(entry)
    mo.vstack(
        [
            mo.md(
                "Top-1 grasp success (%). Retrained grounders differ slightly between runs "
                "(non-deterministic GPU kernels); released checkpoints reproduce the paper."
            ),
            mo.ui.table(rows, selection=None),
        ]
    )
    return (run_ids,)


@app.cell
def _(mo):
    gallery_button = mo.ui.run_button(label="5. Gallery")
    gallery_button
    return (gallery_button,)


@app.cell
def _(R, g2g, gallery_button, mo, run_ids, ws):
    mo.stop(not (gallery_button.value and run_ids), mo.md("Press **5. Gallery**."))
    sheets = []
    for _run_id in ("ours-full-seen-01", "ours-full-unseen-01"):
        if not (ws.gallery / _run_id).exists():
            g2g(R.gallery_command(ws, _run_id), "gallery-" + _run_id)
        sheets.append(
            mo.vstack(
                [mo.md(f"**{_run_id}**"), mo.image(ws.gallery / _run_id / "contact_sheet.png")]
            )
        )
    mo.vstack(
        [
            mo.md(
                "Green: ground truth for the instruction. Red: our top-1 grasp. Seeded mix of successes and failures."
            ),
            *sheets,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
