# Implementation validation — 2026-10-26

- `python -m pytest -q`: **35 passed** locally, including the pinned LGD
  comparison on 200 seeded rectangle pairs, real tiny CLIPSeg shape/gradient
  checks (no weight download), deterministic resume, concurrent ZIP cache and
  registry writers, and Windows checkpoint-failure preservation/retry.
- `python -m ruff check g2g tests`: passed.
- `python -m marimo check` on the four external notebooks: passed static checks.
  Local marimo warns that the global user-config directory cannot be read in
  the sandbox; there are no notebook syntax/dependency-graph errors.
- Real schema verification: 20 matched image/label/mask/instruction samples;
  10 overlays rendered and visually reviewed. Evidence is in this directory.
- `g2g validate-data` on the complete smoke manifest: 49 rows, 14 images,
  zero failures; see `smoke_data_validation.json`.
- Real CPU train/eval/checkpoint/registry flow completed. See `EXPERIMENTS.md`
  for run IDs and the explicit limitation of the reduced overfit diagnostic.

The notebooks remain outside the Git repository. A successful static check is
not a molab execution, GPU compatibility test, or pretrained CLIPSeg training
result. Full-capacity overfit, R1/R2 benchmarks and the paper remain pending.
