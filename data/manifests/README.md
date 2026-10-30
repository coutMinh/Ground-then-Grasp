# Manifests

`fullpool-8144/` is the split used for every reported number. It was built with
`python -m g2g build-manifests --train-images 8144 --val-images 200 --test-images 0`
(seed 42) from the official Grasp-Anything++ object lists at the dataset revisions
recorded in `protocol.json`:

| Split | Images | Rows (image, instruction) |
|---|---|---|
| train | 8,144 | 27,870 |
| val (held out from train scenes) | 200 | 662 |
| test_seen (full official list) | 412 | 1,376 |
| test_unseen (full official list) | 193 | 601 |

No image appears in two splits. `protocol.json` stores each file's SHA-256;
`validation.json` reports zero decode/parse failures over all rows. `all.csv`
(the concatenation used by `download-subset`) is not committed; the notebook
rebuilds it and checks the hashes.
