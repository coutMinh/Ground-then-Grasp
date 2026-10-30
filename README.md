# Ground-then-Grasp (G2G)

Language-driven grasp detection on [Grasp-Anything++](https://github.com/Fsoft-AIC/Grasp-Anything):
given an RGB image and an instruction such as *"Grasp the mug handle"*, predict one
rectangle grasp `g = (x, y, w, h, θ)`.

**Checkpoints:** [huggingface.co/coutMinh/ground-then-grasp](https://huggingface.co/coutMinh/ground-then-grasp)

![G2G overview](docs/figs/pipeline.png)

G2G splits the task into two parts, trained separately and combined only at inference:

- **Grasp prior** (GR-ConvNet, 1.90 M params, RGB only, never sees text). It predicts dense
  grasp quality `Q`, angle `(cos 2θ, sin 2θ)` and width on a 224 × 224 grid. Its targets
  are the union of every annotated grasp in the scene.
- **Part grounder** (CLIPSeg `clipseg-rd64-refined`). The CLIP encoders stay frozen; only
  the 1.13 M-param decoder is fine-tuned, with BCE + Dice loss on GA++ part masks.
  Its output is a part map `P = σ(z / τ)`.
- **Fusion.** `F = Q^α · P^(1−α)`. After Gaussian smoothing, up to 10 peaks are taken,
  scored by `log F + λ · log(mean P inside the rectangle)`, and the best one is returned.
  `α`, `τ` and `λ` are chosen on val only.

## Results

Top-1 grasp success (%) is the GA++ / LGD metric: rotated IoU > 0.25 with some ground-truth
grasp for that instruction, and angle error ≤ 30°. All rows share the same prior
checkpoint and test manifests (one seed). *Pointing* is the percentage of predicted
grasp centres that lie inside the ground-truth part mask.

| System | Seen | Unseen | H | Pointing S / U |
|---|---|---|---|---|
| Prior only (no text) | 33.6 | 32.3 | 32.9 | 31.2 / 40.3 |
| Mask filter, zero-shot CLIPSeg | 32.0 | 33.3 | 32.7 | 48.0 / 66.4 |
| Mask filter, fine-tuned CLIPSeg | 36.4 | 33.4 | 34.9 | 65.0 / 68.7 |
| Mask peak | 18.4 | 13.8 | 15.8 | 71.9 / 71.4 |
| **G2G (ours)** | **37.6** | **35.8** | **36.7** | 61.0 / 66.4 |
| Oracle: ground-truth part mask + prior *(diagnostic)* | 46.6 | 41.6 | 43.9 | 100 / 100 |
| GR-ConvNet + CLIP † | 37 | 18 | 24 | – |
| LGD † | 48 | 42 | 45 | – |

† Numbers published in [LGD (CVPR 2024)](https://arxiv.org/abs/2406.09489), Table 2. They come
from full training data and a different protocol, so they are not directly comparable.

- **Setup:** 8,144 train images (27,870 instructions); 200 val images (662). The tests
  are the full official Seen list (412 images, 1,376 instructions) and Unseen list
  (193 images, 601 instructions).
- **Selected composition:** α = 0.75, τ = 2, λ = 0. Because λ = 0, re-ranking is inactive
  and G2G equals plain soft fusion. With α < 1, raising λ lowered val success.
- **Grounder mIoU on test:** fine-tuned 0.555 / 0.572; zero-shot 0.369 / 0.467.
- **Where the files are:** every number traces to
  [`results/fullpool-8144/`](results/fullpool-8144/) (registry, per-run metrics and
  predictions, generated [`main.csv`](results/fullpool-8144/tables/main.csv)).

### Qualitative examples

Green: ground-truth grasps for the instruction. Red: our top-1 prediction. The examples
are a seeded mix of successes and failures (see `selection_*.json`), so they are not an
accuracy estimate.

| Seen ✓ *"Take hold of mug on its handle."* | Seen ✗ *"Pick up mug by its handle."* |
|---|---|
| ![](docs/figs/gallery/seen_success_mug_handle.png) | ![](docs/figs/gallery/seen_failure_mug_handle.png) |
| **Unseen ✓** *"Grasp lamp at its stand."* | **Unseen ✗** *"Grip hairbrush on its handle."* |
| ![](docs/figs/gallery/unseen_success_lamp_stand.png) | ![](docs/figs/gallery/unseen_failure_hairbrush_handle.png) |

More examples: [Seen contact sheet](docs/figs/gallery/contact_sheet_seen.png) ·
[Unseen contact sheet](docs/figs/gallery/contact_sheet_unseen.png).

## Repository layout

```text
g2g/
  data/       remote ZIP reader, manifests, selective download, GA++ parsing, grasp maps
  models/     GR-ConvNet prior (adapted from LGD, MIT), CLIPSeg grounder wrapper
  compose.py  fusion, peak finding, re-ranking (Eq. F = Q^α P^(1−α))
  train.py    one trainer for both experts (AMP, resume, registry row per run)
  eval/       rotated-IoU metric, evaluation of all systems, val sweep, tables
  viz/        prediction-vs-ground-truth figures
  reproduce.py  the exact command sequence behind the reported results
  cli.py      `python -m g2g <command>`
configs/      prior.yaml (R1), grounder.yaml (R2)
data/manifests/fullpool-8144/   the split used for every number (with SHA-256)
notebooks/reproduce_molab.py    one marimo notebook: setup → data → checkpoints → eval → gallery
results/fullpool-8144/          registry, per-run metrics/predictions, generated table
docs/         data schema notes, real-data verification, figures
tests/        CPU unit tests
```

## Install

Requires Python 3.11–3.13. Training needs a CUDA GPU; the reported runs used one
NVIDIA RTX PRO 6000 (Blackwell) with PyTorch 2.11 / CUDA 13.0.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[grounder,dev]"
python -m pytest -q
```

## Data access

Accept the terms of the dataset owners
([download page](https://airvlab.github.io/grasp-anything/docs/download/); HF repos
`airvlab/Grasp-Anything` and `airvlab/Grasp-Anything-pp`). Then export `HF_TOKEN` (read
scope) in your shell or as a molab secret. The code reads the token only from the
environment and never logs it. The dataset is not redistributed here.

`download-subset` does not download the ~65 GB image archives. It reads the ZIP
central directories over HTTP Range requests and extracts only the manifest members
(about 9k images).

## Reproduce on molab

Upload [`notebooks/reproduce_molab.py`](notebooks/reproduce_molab.py) to a GPU molab
session, set the `HF_TOKEN` secret, and press the five buttons in order.

| Step | What it does | Time |
|---|---|---|
| 1. Setup | clone this repo, `pip install -e`, GPU probe, `pytest`, pinned LGD split lists | ~5 min |
| 2. Data | build `fullpool-8144` manifests, check them against the committed SHA-256, download, validate | 1–1.5 h |
| 3. Checkpoints | **HF mode** (default): download and hash-check the released checkpoints. **Train mode**: R1 prior (~25 min) ∥ R2 grounder (~65 min), then val sweep (~10 min) | 1 min / ~75 min |
| 4. Evaluate | 7 systems × Seen/Unseen, `make-tables`, side-by-side with the committed table | ~12 min |
| 5. Gallery | draw saved predictions against ground truth | ~1 min |

HF mode should reproduce the table above. Training again gives close but not
bit-identical numbers, because the grounder uses non-deterministic GPU kernels.

The same steps from a terminal (paths are examples; see `g2g/reproduce.py` for the
exact arguments):

```bash
python -m g2g build-manifests --upstream LGD --output manifests --cache-dir cache \
    --train-images 8144 --val-images 200 --test-images 0
python -m g2g download-subset --manifest manifests/all.csv --data-root data --cache-dir cache
python -m g2g train --config configs/prior.yaml    --manifest manifests/train.csv \
    --val-manifest manifests/val.csv --data-root data --results-root results \
    --run-id prior-fullpool-01 --purpose benchmark
python -m g2g train --config configs/grounder.yaml ...  --run-id grounder-fullpool-01
python -m g2g sweep --manifest manifests/val.csv --prior-checkpoint P --grounder-checkpoint G ...
python -m g2g evaluate --manifest manifests/test_seen.csv --system ours \
    --prior-checkpoint P --grounder-checkpoint G --compose-config best_compose.json ...
python -m g2g make-tables --results-root results --run-ids <eval runs> --output tables
python -m g2g visualize-predictions --run-dir results/runs/ours-full-seen-01 --data-root data --output gallery
```

## Checkpoints

| File on [HF](https://huggingface.co/coutMinh/ground-then-grasp) | Content | SHA-256 |
|---|---|---|
| `prior/best.pt` | R1 prior, best val epoch (15 epochs) | `d5089305…` |
| `grounder/best.pt` | R2 grounder, best val mIoU 0.578 (epoch 8 of 10), full state | `ec5d4c3f…` |
| `grounder/decoder_only.pt` | decoder weights only (1,127,009 params, key `decoder_state`) | `b8e2e422…` |
| `composition/best_compose.json` | α, τ, λ chosen on val, plus the two checkpoint hashes | – |

`evaluate` checks that `best_compose.json` matches the checkpoint hashes. Use the full
`best.pt` files with `g2g.eval.evaluate.load_expert`; `decoder_only.pt` needs the
pretrained CLIPSeg weights to be loaded first.

## Limitations

- One seed.
- 8,144 training images; the pinned dataset release has only 8,344 eligible train-pool images.
- Synthetic images, 2D rectangles, one grasp per instruction.
- No retrained joint baseline (e.g. LGR-ConvNet + CLIP) and no official LGD checkpoint
  under our protocol.
- Some GA++ part masks and grasps disagree with their instruction (see
  [REAL_DATA_VERIFICATION.md](docs/REAL_DATA_VERIFICATION.md)). The oracle row is
  therefore not an upper bound on grounding quality.

## Acknowledgements and license

- The GR-ConvNet prior and the metric conventions follow the
  [LGD repository](https://github.com/Fsoft-AIC/LGD) (MIT; see [THIRD_PARTY.md](THIRD_PARTY.md)).
- The grounder uses [CLIPSeg](https://huggingface.co/CIDAS/clipseg-rd64-refined).
- Code: MIT ([LICENSE](LICENSE)).
