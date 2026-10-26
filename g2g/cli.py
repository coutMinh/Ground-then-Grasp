"""CLI commands import heavy dependencies only when needed."""

from __future__ import annotations

import argparse
import json


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="g2g", description="Ground First, Then Grasp")
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe_parser = subparsers.add_parser("probe", help="Save a CPU/GPU environment report")
    probe_parser.add_argument("--output", required=True)
    audit_parser = subparsers.add_parser("audit-upstream", help="Audit pinned LGD split metadata")
    audit_parser.add_argument("--root", required=True)
    audit_parser.add_argument("--output", required=True)
    access_parser = subparsers.add_parser(
        "check-access", help="Check HF API/HEAD, without downloads"
    )
    access_parser.add_argument("--output", required=True)
    inspect_parser = subparsers.add_parser("inspect-files", help="Describe supplied local samples")
    inspect_parser.add_argument("--paths", nargs="+", required=True)
    inspect_parser.add_argument("--output", required=True)
    archive_parser = subparsers.add_parser(
        "inspect-archive", help="Read ZIP inventory via HTTP Range"
    )
    archive_parser.add_argument("--urls", nargs="+", required=True)
    archive_parser.add_argument("--output", required=True)
    archive_parser.add_argument("--members", nargs="*")
    archive_parser.add_argument("--destination")
    archive_parser.add_argument("--limit", type=int, default=20)
    samples_parser = subparsers.add_parser(
        "verify-samples", help="Fetch matched real samples for VERIFY"
    )
    samples_parser.add_argument("--upstream", required=True)
    samples_parser.add_argument("--destination", required=True)
    samples_parser.add_argument("--count", type=int, default=20)
    overlays_parser = subparsers.add_parser(
        "schema-overlays", help="Parse real samples and draw overlays"
    )
    overlays_parser.add_argument("--data-root", required=True)
    overlays_parser.add_argument("--output", required=True)
    overlays_parser.add_argument("--limit", type=int, default=10)
    manifest_parser = subparsers.add_parser(
        "build-manifests", help="Build image-disjoint official subsets"
    )
    manifest_parser.add_argument("--upstream", required=True)
    manifest_parser.add_argument("--output", required=True)
    manifest_parser.add_argument("--cache-dir", required=True)
    manifest_parser.add_argument("--train-images", type=int, default=100)
    manifest_parser.add_argument("--val-images", type=int, default=20)
    manifest_parser.add_argument("--test-images", type=int, default=20)
    manifest_parser.add_argument("--seed", type=int, default=42)
    download_parser = subparsers.add_parser("download-subset", help="Extract only manifest members")
    download_parser.add_argument("--manifest", required=True)
    download_parser.add_argument("--data-root", required=True)
    download_parser.add_argument("--cache-dir", required=True)
    download_parser.add_argument("--workers", type=int, default=8)
    validate_parser = subparsers.add_parser(
        "validate-data", help="Decode and validate all manifest rows"
    )
    validate_parser.add_argument("--manifest", required=True)
    validate_parser.add_argument("--data-root", required=True)
    validate_parser.add_argument("--output", required=True)
    train_parser = subparsers.add_parser("train", help="Train prior or CLIPSeg decoder")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--manifest", required=True)
    train_parser.add_argument("--val-manifest", required=True)
    train_parser.add_argument("--data-root", required=True)
    train_parser.add_argument("--results-root", required=True)
    train_parser.add_argument("--run-id", required=True)
    train_parser.add_argument("--resume")
    train_parser.add_argument("--purpose", choices=["debug", "benchmark"], default="debug")
    train_parser.add_argument("--stop-after-epochs", type=int)
    eval_parser = subparsers.add_parser("evaluate", help="Top-1 evaluation against native GT")
    eval_parser.add_argument("--manifest", required=True)
    eval_parser.add_argument("--data-root", required=True)
    eval_parser.add_argument("--prior-checkpoint", required=True)
    eval_parser.add_argument("--grounder-checkpoint")
    eval_parser.add_argument("--results-root", required=True)
    eval_parser.add_argument("--run-id", required=True)
    eval_parser.add_argument(
        "--system",
        default="prior",
        choices=["prior", "zero_shot_hard", "ft_hard", "soft", "ours", "mask_peak", "oracle_mask"],
    )
    eval_parser.add_argument("--purpose", choices=["debug", "benchmark"], default="debug")
    eval_parser.add_argument("--alpha", type=float, default=0.5)
    eval_parser.add_argument("--temperature", type=float, default=1)
    eval_parser.add_argument("--rerank-weight", type=float, default=0.5)
    eval_parser.add_argument("--compose-config", help="Frozen best_compose.json produced on val")
    eval_parser.add_argument(
        "--union-targets", action="store_true", help="Prior-only debug overfit check"
    )
    sweep_parser = subparsers.add_parser(
        "sweep", help="Choose composition on val; forward experts once"
    )
    sweep_parser.add_argument("--manifest", required=True)
    sweep_parser.add_argument("--data-root", required=True)
    sweep_parser.add_argument("--prior-checkpoint", required=True)
    sweep_parser.add_argument("--grounder-checkpoint", required=True)
    sweep_parser.add_argument("--results-root", required=True)
    sweep_parser.add_argument("--run-id", required=True)
    sweep_parser.add_argument("--purpose", choices=["debug", "benchmark"], default="debug")
    tables_parser = subparsers.add_parser(
        "make-tables", help="Generate tables from selected benchmark runs"
    )
    tables_parser.add_argument("--results-root", required=True)
    tables_parser.add_argument("--run-ids", nargs="+", required=True)
    tables_parser.add_argument("--output", required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "probe":
        from g2g.utils.environment import probe

        report = probe(arguments.output)
    elif arguments.command == "audit-upstream":
        from g2g.inspection.upstream import audit_upstream

        report = audit_upstream(arguments.root, arguments.output)
    elif arguments.command == "check-access":
        from g2g.inspection.access import check_access

        report = check_access(arguments.output)
    elif arguments.command == "inspect-files":
        from g2g.inspection.files import inspect_files

        report = inspect_files(arguments.paths, arguments.output)
    elif arguments.command == "inspect-archive":
        from g2g.inspection.archives import inspect_archive

        report = inspect_archive(
            arguments.urls,
            arguments.output,
            members=arguments.members,
            destination=arguments.destination,
            limit=arguments.limit,
        )
    elif arguments.command == "verify-samples":
        from g2g.inspection.samples import fetch_verification_samples

        report = fetch_verification_samples(
            arguments.upstream, arguments.destination, arguments.count
        )
    elif arguments.command == "schema-overlays":
        from g2g.viz.schema import schema_overlays

        report = schema_overlays(arguments.data_root, arguments.output, arguments.limit)
    elif arguments.command == "build-manifests":
        from g2g.data.manifest import build_manifests

        report = build_manifests(
            arguments.upstream,
            arguments.output,
            arguments.cache_dir,
            train_images=arguments.train_images,
            val_images=arguments.val_images,
            test_images=arguments.test_images,
            seed=arguments.seed,
        )
    elif arguments.command == "download-subset":
        from g2g.data.download import download_subset

        report = download_subset(
            arguments.manifest, arguments.data_root, arguments.cache_dir, arguments.workers
        )
    elif arguments.command == "train":
        from g2g.train import train

        report = train(
            arguments.config,
            arguments.manifest,
            arguments.val_manifest,
            arguments.data_root,
            arguments.results_root,
            arguments.run_id,
            resume=arguments.resume,
            purpose=arguments.purpose,
            stop_after_epochs=arguments.stop_after_epochs,
        )
    elif arguments.command == "validate-data":
        from g2g.inspection.validate import validate_data

        report = validate_data(arguments.manifest, arguments.data_root, arguments.output)
    elif arguments.command == "evaluate":
        from g2g.eval.evaluate import evaluate

        report = evaluate(
            arguments.manifest,
            arguments.data_root,
            arguments.prior_checkpoint,
            arguments.results_root,
            arguments.run_id,
            system=arguments.system,
            grounder_checkpoint=arguments.grounder_checkpoint,
            purpose=arguments.purpose,
            alpha=arguments.alpha,
            temperature=arguments.temperature,
            rerank_weight=arguments.rerank_weight,
            compose_config=arguments.compose_config,
            union_targets=arguments.union_targets,
        )
    elif arguments.command == "make-tables":
        from g2g.eval.tables import make_tables

        report = make_tables(arguments.results_root, arguments.run_ids, arguments.output)
    elif arguments.command == "sweep":
        from g2g.eval.sweep import sweep

        report = sweep(
            arguments.manifest,
            arguments.data_root,
            arguments.prior_checkpoint,
            arguments.grounder_checkpoint,
            arguments.results_root,
            arguments.run_id,
            purpose=arguments.purpose,
        )
    if arguments.command == "download-subset":
        report = {
            "complete": report["complete"],
            "counts": {name: value["count"] for name, value in report["stages"].items()},
        }
    elif arguments.command == "verify-samples":
        report = {"sample_count": len(report["sample_ids"]), "destination": arguments.destination}
    print(json.dumps(report, indent=2))
