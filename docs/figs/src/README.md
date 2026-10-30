# Compact G2G figure for a single CVPR column

The figure retains its 760 × 376 layout and prints at 8.26 × 4.08 cm in a
3.25 inch CVPR column. The three post-processing blocks have equal sizes,
equal gaps, and centered labels.

Source files for `../pipeline.png` (rendered from the PDF at 300 dpi), which is
the paper's Fig. 1:

- `g2g_pipeline_compact.drawio`: editable native vector objects and text.
- `g2g_pipeline_compact.pdf`: vector figure for sharp LaTeX output.
- `figure.tex`: single-column insertion snippet and caption.
- `README.md`: this file.

Open the diagram in draw.io Desktop or diagrams.net. Double-click text to
edit it. The geometry icons and post-processing thumbnails are grouped so
each illustration can be moved as a unit; ungroup to edit individual vector
elements. There are no external image dependencies. The PNG, PDF, and
draw.io diagram share one scene graph.

Blue denotes frozen encoders, coral denotes trainable weights during the
corresponding independent training run, and green denotes fixed operations.
Red loss annotations describe training. The experts are both evaluated
without training during inference; fusion and post-processing introduce no
learned weights. Alpha, tau, and lambda are selected on validation and then
fixed for testing.

The user's edited title **G2G**, short **Part mask** label, and removal of
the loss formulas and short training arrows are preserved. **Union grasp
poses (train)** explicitly denotes supervision, not an inference input.
The prior produces **Quality Q** and separate **Angle / width** maps.
The grounder produces **Part map P**. Soft PoE shows the fusion equation
followed by its actual output heatmap **F**. Q/P/F use exactly the same
scene coordinates and fixed 0..1 color scale; F is computed as sqrt(Q*P)
in the illustrative example. Grasp geometry continues from the prior.

The Gaussian thumbnails show the same full scene before and after an actual
Gaussian filter, using the same color scale. Sharp isolated noise is
attenuated and the broader mug-handle support remains. There are no peak
markers at this stage, and smoothing does not imply that every weak local
maximum disappears. Five peak/rectangle candidates are shown next: three
near the handle, one on the mug body, and a weaker one near the spoon.

Reranking compares A (higher center score, poor inner-part coverage),
B (slightly lower center score, better coverage, selected in green), and
C (weak spoon candidate with low part probability). Small horizontal bars
show relative final scores, computed from the synthetic smoothed F and
mean P inside each half-size rectangle, with illustrative lambda=1.
The bars are exp(score) normalized to the winner; no numerical benchmark
results are claimed. In the score formula, F refers to the smoothed center
score. The final green grasp is the same candidate B. The same mug/spoon
geometry is used in every panel; all examples are explicitly schematic.

Implementation details retained in the paper's Method section:

- The GR-ConvNet prior trains on the union of permitted positive grasp
  annotations per image, using four SmoothL1 losses.
- CLIPSeg's CLIP encoders are frozen; the FiLM/transformer decoder is
  fine-tuned with BCE-with-logits plus soft Dice against the part mask.
- `Q = clip(q, 0, 1)`; `P = sigmoid(z / tau)`. Angle components are
  `c = cos(2 theta)` and `s = sin(2 theta)`; `u` is normalized width.
- Gaussian sigma is 2 for F/c/s and 1 for u. P remains available for
  reranking and is not Gaussian-smoothed.
- Up to ten local peaks have minimum distance 5 grid pixels. Decoding uses
  `theta = atan2(s,c)/2`, `w = 112 clip(u,0,1)`, and `h = w/2`.
- Ranking adds log smoothed center quality to lambda times log mean P in
  the rectangle shrunk by 0.5 along both dimensions, with epsilon clamping.
  A valid top-1 grasp is rescaled to native coordinates; no valid candidate
  yields no prediction.

These semantics were checked against `g2g/data/dataset.py`,
`g2g/models/grounder.py`, `g2g/models/grconvnet3.py`, `g2g/losses.py`,
`g2g/compose.py`, and the paper's `sec/3_method.tex` and `main.pdf`.

Regenerate from the repository root with:

```powershell
python scripts/draw_compact_pipeline.py
```

This needs Pillow and PyMuPDF. Arial is located through `WINDIR` on Windows;
DejaVu Sans is used elsewhere. Layout checks are printed to stdout, and
only the three final exports are written. Diagnostic previews belong in a
temporary directory. The original large figure and paper files are preserved.
