"""The notebook's command sequence must match the committed run-2 configs."""

import json
from pathlib import Path

from g2g import reproduce as R
from g2g.cli import build_parser

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "results" / R.MANIFEST_SET / "runs"
COMPOSE = RUNS / R.SWEEP_RUN / "best_compose.json"


def test_every_command_parses():
    ws = R.Workspace(Path("/workspace"))
    prior, grounder = Path("/p.pt"), Path("/g.pt")
    commands = [arguments for _, arguments in R.data_commands(ws)]
    commands += [R.train_command(ws, config, run_id) for config, run_id in R.TRAIN_RUNS.values()]
    commands += [R.sweep_command(ws, prior, grounder)]
    evaluations = R.evaluation_commands(ws, prior, grounder, COMPOSE)
    commands += [arguments for _, arguments in evaluations]
    commands += [R.table_command(ws, [run_id for run_id, _ in evaluations])]
    commands += [R.gallery_command(ws, "ours-full-seen-01")]
    parser = build_parser()
    for arguments in commands:
        parser.parse_args(arguments)
    for config, _ in R.TRAIN_RUNS.values():
        assert (REPO / "configs" / config).is_file()


def test_evaluations_match_committed_run_configs():
    ws = R.Workspace(Path("/workspace"))
    parser = build_parser()
    evaluations = R.evaluation_commands(ws, Path("/p.pt"), Path("/g.pt"), COMPOSE)
    assert {run_id for run_id, _ in evaluations} == {
        path.name for path in RUNS.iterdir() if path.name.endswith(("-seen-01", "-unseen-01"))
    }
    frozen = json.loads(COMPOSE.read_text(encoding="utf-8"))["composition"]
    for run_id, arguments in evaluations:
        args = parser.parse_args(arguments)
        saved = json.loads((RUNS / run_id / "config.json").read_text(encoding="utf-8"))
        assert args.system == saved["system"]
        assert Path(args.manifest).stem == saved["split"]
        assert (args.grounder_checkpoint is None) == (saved["grounder_checkpoint"] is None)
        used = (
            frozen
            if args.compose_config
            else {
                "alpha": args.alpha,
                "temperature": args.temperature,
                "rerank_weight": args.rerank_weight,
            }
        )
        for key in ("alpha", "temperature", "rerank_weight"):
            assert float(used[key]) == float(saved["composition"][key]), (run_id, key)


def test_committed_manifests_match_protocol_hashes():
    manifests = REPO / "data/manifests" / R.MANIFEST_SET
    assert all(R.check_manifests(manifests, manifests / "protocol.json").values())
