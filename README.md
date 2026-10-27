# Ground-then-Grasp

Factorized language-driven 2D grasp detection: a language-free GR-ConvNet prior
is combined with a CLIPSeg part grounder through soft quality/mask composition
and optional part-aware re-ranking. This is a QACI homework implementation.
Accuracy claims require completed, registered experiments; none are inferred
from tests or from published LGD numbers.

## Repository and molab layout

This directory is the standalone GitHub repository:
<https://github.com/coutMinh/Ground-then-Grasp>.

**Molab notebooks are distributed separately** in sibling `../molab-notebooks/`.
They clone this repository; they are intentionally not part of this checkout.
On molab:

```text
workspace/
  notebooks/                  # uploaded marimo files
  repos/ground-then-grasp/     # cloned code
  repos/LGD/                  # pinned upstream reference, unmodified
  artifacts/                  # data, ZIP cache, checkpoints, registry, reports
```

Upload sibling `g2g_pipeline.py` to run setup, data, training and evaluation
in one runtime. The four numbered notebooks remain as references; separate
molab runtimes do not imply shared storage. Model and dataset logic stays
in `g2g/`. Publish the code to GitHub before the clone-based workflow.

To inspect saved predictions, form 6 draws GT/prediction/overlay panels and
prints instructions, native-coordinate grasps and IoU. For an already running
cloud notebook, paste sibling `view_predictions_cell.py` into one new cell.
Both cells invoke `python -m g2g visualize-predictions`; rendering lives in
`g2g/viz/predictions.py`. Publish this code to GitHub first. Enable the form's
checkout-update checkbox once, or set `_update_checkout = True` in the pasted
cell, to fetch a published branch/tag/SHA. The update refuses local edits and
checks for the module before switching commits; artifacts stay outside the repo.

## Install and checks

Python 3.11–3.13. Local CPU development supports torch >=2.3; pretrained-model
training on the intended Blackwell GPU uses a working CUDA 12.8 PyTorch >=2.7
installation. Notebook 00 offers an explicit torch 2.7.1 cu128 install option.
See [official PyTorch notes](https://pytorch.org/blog/pytorch-2-7/) and the
[pinned CLIPSeg API documentation](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/clipseg).

```bash
python -m venv .venv
# Activate .venv using your shell, then:
python -m pip install -e ".[dev,grounder]"
python -m pytest -q
python -m ruff check g2g tests
python -m g2g --help
```

The current Windows development venv reuses preinstalled CPU torch through
`--system-site-packages` to avoid another large download. It is ignored by Git;
this is not required on molab. Tests construct a tiny CLIPSeg configuration
without downloading pretrained weights, when the optional dependency is installed.

## Dataset verification

Complete the owner's [download form/license](https://airvlab.github.io/grasp-anything/docs/download/).
Keep any `HF_TOKEN` in environment/molab secrets. `.env.example` documents
variables; no `.env` is loaded implicitly and tokens are never CLI arguments.
The 2026-10-26 access check found public HTTP access; owner requirements still apply.

The pinned reference clone is outside this repository:

```bash
git clone https://github.com/Fsoft-AIC/LGD.git ../third_party/LGD
git -C ../third_party/LGD checkout --detach 501904dc5c30c8fad58d90f7e4f49b4095f11eb0
python -m g2g audit-upstream --root ../third_party/LGD --output ../artifacts/upstream_splits.json
python -m g2g verify-samples --upstream ../third_party/LGD --destination ../artifacts/verify_samples
python -m g2g schema-overlays --data-root ../artifacts/verify_samples --output ../artifacts/schema_overlays
```

See [DATA_SCHEMA.md](docs/DATA_SCHEMA.md) and
[REAL_DATA_VERIFICATION.md](docs/REAL_DATA_VERIFICATION.md) for code references,
real-file inspection, label noise, and the reviewed contact sheet.

Archives contain millions of files. ZIP central-directory reads themselves
transfer gigabytes; cached directories are stored outside the checkout. Member
extraction uses bounded HTTP Range, ZIP64 support, retries, CRC and SHA-256.
Workers share selected offsets instead of each parsing the complete ZIP index.
If the host rejects Range or lacks RAM/disk, use full local archives on an
appropriately sized GPU machine; that fallback is not automatically downloaded.

## Manifests and training

Start with the checked-in `data/manifests/smoke/` manifests for a small CPU
pipeline check. They preserve all actual part rows for permitted objects on
8 train, 2 val, 2 Seen-test and 2 Unseen-test images. They are not final paper splits.
To create a larger pilot:

```bash
python -m g2g build-manifests --upstream ../third_party/LGD --output ../artifacts/manifests/pilot --cache-dir ../artifacts/inspection/http-cache --train-images 100 --val-images 20 --test-images 20
python -m g2g download-subset --manifest ../artifacts/manifests/pilot/all.csv --data-root ../artifacts/ga_subset --cache-dir ../artifacts/inspection/http-cache --workers 8
```

Manifests use official **object** IDs and enumerate actual part files. All test
images are excluded before sampling train/val. The official train list contains
37 images also in test Unseen; our stricter protocol removes them. The released
positive archive has labels for 8,344 eligible train-pool images in this audit,
before validation holdout. Requested counts exceeding availability fail explicitly.
Use `--test-images 0` for all available official test images. Text is read from
the native `instruction_member`; it is not guessed from object descriptions.

```bash
python -m g2g train --config configs/prior_smoke.yaml --manifest data/manifests/smoke/train.csv --val-manifest data/manifests/smoke/val.csv --data-root ../artifacts/ga_smoke --results-root ../artifacts/results --run-id prior-smoke --purpose debug
```

For real R1/R2 use `configs/prior.yaml` / `configs/grounder.yaml`, pilot/final
manifests, and explicit `--purpose benchmark` after GPU and overfit checks.
There are only two mandatory trainings. Prior training samples each image once,
using union grasps from rows permitted in that split. Grounder trains only the
decoder; frozen CLIP remains in evaluation mode. No geometric text augmentation
is applied. Warmup is capped to 10% of the actual step budget for tiny runs.

`--resume <previous-run>/last.pt` starts a new run directory and restores model,
optimizer, scheduler, AMP scaler, epoch/global step, RNG and dataloader generator.
Keep sibling `best.pt` to preserve best-model selection. Config and train/val
manifest hashes must match. Resume is at epoch boundaries; no mid-epoch promise.
Use a new run ID; existing run directories are never overwritten.

`configs/prior_overfit.yaml` provides a longer small-data memorization diagnostic.
Evaluate its `last.pt` against the train manifest with `--system prior --union-targets
--purpose debug` to measure scene-union training success. This diagnostic samples
each image once and omits instruction pointing; it cannot be labeled a benchmark.

## Evaluation and comparison protocol

```bash
python -m g2g evaluate --manifest data/manifests/smoke/test_seen.csv --data-root ../artifacts/ga_smoke --prior-checkpoint ../artifacts/results/runs/prior-smoke/best.pt --results-root ../artifacts/results --run-id prior-smoke-seen --system prior --purpose debug
```

Systems: `prior`, `zero_shot_hard`, `ft_hard`, `soft` (no re-ranking), `ours`,
`mask_peak`, `oracle_mask`. Fine-tuned systems require `--grounder-checkpoint`.
`zero_shot_hard` downloads the original CLIPSeg weights. Configure `--alpha`,
`--temperature`, `--rerank-weight` on **val**, then freeze before test.

`g2g sweep` forwards each expert once per val sample and evaluates the fixed
alpha/temperature/re-ranking grid. It rejects train/test manifests and writes
`best_compose.json`. Pass that file to `evaluate --compose-config ...`; checkpoint
hashes must match. `g2g make-tables --results-root ... --run-ids ... --output ...`
generates CSV/LaTeX with provenance from explicitly selected, paired Seen/Unseen
benchmark runs, rejects debug runs and incompatible test manifests, and computes
H per seed before aggregating. No published-paper scores are injected by code.

Top-1 success uses native-coordinate pixel IoU **>0.25** and angle difference
**<=30 degrees**, modulo pi. These boundary choices follow the actual LGD code.
Predictions, angle-eligible best IoU, success and native-mask pointing are saved
per instruction. Missing predictions count as failures; empty GT fails validation.
CLIPSeg mIoU uses the working map resolution; empty predicted/GT union scores 1.
Our consistent width scale and circular smoothing are documented adaptations.

Every train/eval invocation records config, manifest hash, commit/dirty flag,
seed, hardware, duration and metrics in the chosen results registry. `debug`
runs are pipeline checks and must not populate paper tables. The main comparison
uses matched self-run ablations; published LGD/baseline scores belong in a separate
reported-results block with their own data/protocol and verified citation.

## Current boundary

Code/real-data checks and CPU smoke are distinct from GPU experiments. Consult
`docs/EXPERIMENTS.md`, `docs/BLOCKERS.md` and the external `../plan_fixed.md` for
actual completion. Final-scale R1/R2, val tuning, full test results, paper tables,
figures and the two-page CVPR submission remain gated on their recorded checks.
No accuracy or runtime estimate in the original plan is a measured result.
