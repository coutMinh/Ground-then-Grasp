# Results

`fullpool-8144/` holds every artifact behind the paper's numbers (run 2, single
seed). Checkpoints are not stored here; they are on
[Hugging Face](https://huggingface.co/coutMinh/ground-then-grasp).

- `registry.csv`: one row per run (git commit `9929373`, config, manifest
  SHA-256, seed, GPU, wall-clock, metrics path). Paths point to the molab
  workspace where the runs were executed.
- `runs/<run_id>/`: `config.*`, `metrics.json` and `log.csv` for training;
  `preds.jsonl` (one top-1 rectangle per test row) for evaluation;
  `best_compose.json` for the val sweep.
- `tables/`: `main.csv` / `main.tex`, generated with `python -m g2g make-tables`
  from the 14 benchmark evaluation runs, plus `provenance.json`.

Training and evaluation were run on molab (one NVIDIA RTX PRO 6000 Blackwell GPU) with
`notebooks/reproduce_molab.py`'s command sequence. New runs append to the
registry under the `--results-root` you pass to the CLI.
