# Experiment log

## 2026-10-26 — implementation started

- Scope: two mandatory trainings (prior and CLIPSeg decoder); joint LGD/B2 optional.
- Notebooks live in sibling `../molab-notebooks/` and clone the codebase from GitHub.
- No real-data experiments or GPU measurements yet. Synthetic checks must be
  tagged `synthetic`; they cannot populate paper result tables.

## 2026-10-26 — verified data and CPU pipeline

- Source and split audit: `DATA_SCHEMA.md`, `upstream_splits.json`.
- Twenty matched real samples parsed; ten overlays viewed. Annotation noise
  is recorded in `REAL_DATA_VERIFICATION.md` and `overlay_verification.json`.
- Smoke manifests: 8 train / 2 val / 2 Seen-test / 2 Unseen-test images;
  49 instruction rows validated with zero failures. All permitted part rows
  are retained when making scene-union targets. Pilot manifests: 160 images,
  532 rows, generated and hashed; not yet downloaded/trained on molab.
- Every actual CPU invocation is recorded under the sibling artifact directory
  `../artifacts/results/registry.csv`. The repository's `results/registry.csv`
  is intentionally still a header: no benchmark experiment has been published.

| Debug run ID | Recorded outcome |
|---|---|
| `prior-cpu-smoke` | Two epochs / four steps completed; checkpoints and registry written |
| `prior-cpu-seen` | Six instruction predictions evaluated; pipeline check only |
| `prior-cpu-unseen` | Six instruction predictions evaluated; pipeline check only |
| `prior-cpu-overfit` | Failed on Windows checkpoint replacement; failure retained in registry |
| `prior-cpu-overfit-resume` | Resumed from last valid epoch-boundary checkpoint, completed 200 total steps |
| `prior-cpu-overfit-union` | Train-union success 0.5 on eight images with the last checkpoint |

The reduced prior uses 96px and 8 base channels in the overfit diagnostic.
Loss fell from 2.033911347389221 to 0.03822926804423332 across the original and
resumed logs. This does **not** pass the planned full-capacity overfit check;
low dense-map loss alone is not a grasp-success acceptance criterion.
Metrics, manifests, configs, logs, predictions and checkpoints remain in
`../artifacts/results/runs/<run-id>/`. All these runs use `purpose=debug` and
are rejected by the paper table generator.

Checkpoint publication now retries transient sharing violations without deleting
the old checkpoint. Registry writes use OS-backed `filelock` locking, tested
with concurrent writers. The old failed run was not rewritten as successful.

## Next gate

Publish the local repository to the supplied GitHub URL, then run external
notebook 00 on molab. Use notebook 01 to create an `overfit` manifest set
(32 train / 4 val images), notebook 02 for `prior_overfit_gpu.yaml` and
`grounder_overfit.yaml`, and notebook 03 for measured train success/mIoU.
Only after this gate should full R1/R2, val tuning and final test comparisons run.
Cloud AMP, pretrained-weight loading/fine-tuning and latency are not verified
by local CPU tests. No paper or final numerical comparison exists yet.
