# LGD code audit and dataset schema

Status: **F1 code inspection and F2 matched real-file verification complete**.
The source-audit sections below retain the distinction between what upstream
code establishes and what was subsequently verified in
[`REAL_DATA_VERIFICATION.md`](REAL_DATA_VERIFICATION.md).
Reference: Fsoft-AIC/LGD commit
`501904dc5c30c8fad58d90f7e4f49b4095f11eb0` (2026-10-26 local inspection).
All source paths/line numbers below refer to that exact commit, not the
moving `main` branch. Base URL:
<https://github.com/Fsoft-AIC/LGD/tree/501904dc5c30c8fad58d90f7e4f49b4095f11eb0>.

## 1. Filename mapping: what the code actually does

Source: `utils/data/grasp_anywhere_data.py:27`, `:30`, `:94–110`, `:123–141`.

- Labels are globbed as `positive_grasp/<image>_<object>_<part>.pt`.
- RGB: `get_rgb` removes `_d_d.pt` with a **single-digit-only regex**, then
  reads `image/<image>.jpg`. A robust adapter must support multi-digit IDs;
  whether such IDs actually exist remains an F2 question.
- Object scene prompt: `prompt/<image>.pkl`; `get_prompts` unpacks
  `(prompt, queries)` and returns **`queries[obj_id]`**.
- Part instruction path is constructed as
  `grasp_instructions/<image>_<object>_<part>.pkl`, but the read is commented
  out at `:138–139`. The published loader **does not read the part instruction**.
  It must not be reused unchanged for instruction-specific part grounding.
- `get_rgb` constructs a candidate `mask/<same-label-stem>.npy` at `:95`,
  but mask loading and image masking are commented out (`:96`, `:101`).
- **Part-mask filename mapping is not established by this loader.** The GA++
  card says `part_mask/*.npy`, but exact archive member names and contents
  require F2. Do not infer verified mapping from a commented-out object mask.
  F2 subsequently verified `part_mask/<same-label-stem>.npy` on 20 matched samples.

## 2. Rectangles, angle and coordinate system

Source: `utils/dataset_processing/grasp.py:19–22`, `:177–196`, `:296–335`,
`:440–472`, `:507–531`; loader resize: `grasp_anywhere_data.py:68–82`.

- `torch.load` is converted to a list; each row is unpacked as
  `(_, x, y, w, h, theta)`. **Six fields**, not the plan's tentative N×5.
  The first field is ignored; its actual meaning/dtype must be inspected.
- External angles are degrees. Adapter converts to `-theta*pi/180`.
- Grasp centre is `[y,x]` in upstream arrays. Coordinates/width/height are
  pixels. Positive internal angle rotates counterclockwise visually; y grows
  downward in the image, so the w-axis direction is `(cos(theta),-sin(theta))`.
- Upstream `length` corresponds to external `w`; upstream `width` corresponds
  to external `h`. Corner ordering retains this distinction.
- Labels are scaled by `output_size/416.0`, then rotated/zoomed.
- `detect_grasps` predicts one dimension only: `g.length=width_map[p]` and
  **`g.width=g.length/2`**. It cannot reconstruct arbitrary independent h.
- For exact geometry round-trips test rectangle<->corners and uniform resize.
  For map decoding test compatible h=w/2 fixtures separately, acknowledging
  clipping and map quantization. Do not require IoU>0.9 for every GT aspect ratio.

## 3. Dense targets and width units

Source: `grasp.py:230–261`, `:344–349`;
`utils/data/language_grasp_data.py:77–96`.

- Q is 1 **through the centre third along w/length**, retaining the full h;
  this is a filled region, not a Gaussian target or just the centre pixel.
- Angle and width are written to the same compact region. Overlaps use the
  **last rectangle in input order**, so union target ordering must be stable.
- Outside grasp regions the angle array is 0, giving cos(2theta)=1 and sin=0.
- Width target: `clip(width_pixels,0,output_size/2)/(output_size/2)`.
  At 224, normalization is **112 pixels**, and larger widths are clipped.
- Upstream `inference/post_process.py:16` instead multiplies by **150**.
  Our decoder will use the same configured normalization as target encoding;
  it will not silently reproduce this 224/150 inconsistency.
- Quality head is linear (`inference/models/grconvnet3.py:36`, `:70–75`),
  SmoothL1 on Q/cos/sin/width (`inference/models/grasp_model.py:16–26`).
  With this loss, Q is a regressed quality, **not a logit**. Composition must
  clamp regressed Q, not sigmoid it (sigmoid(0)=0.5 would brighten background).

## 4. Post-processing and success metric

Source: `inference/post_process.py:14–22`; `grasp.py:351–377`, `:507–531`;
`utils/dataset_processing/evaluation.py:57–79`.

- theta=`atan2(sin,cos)/2`; Gaussian sigma Q=2, theta=2, width=1 pixels.
- Upstream smooths angles directly; our circular smoothing, if enabled, must
  be documented as an adaptation (smooth cos/sin before atan2).
- Upstream peaks: `min_distance=20`, `threshold_abs=0.2`, requested peak count.
  Planned top-K radius=5 is a method choice, not upstream's default.
- `calculate_iou_match` detects grasps, then accepts if any candidate has
  `max_iou(GT)>threshold`. Default threshold 0.25 is **strict >**, despite
  the docstring's greater-or-equal wording.
- Angle difference is `abs((a-b+pi/2)%pi-pi/2)`. Upstream rejects **>pi/6**,
  so equality at 30 degrees is accepted, unlike the plan's strict < wording.
- IoU rasterizes polygons with `skimage.draw.polygon`, not shapely continuous
  areas. Pixel and continuous IoU differ near boundaries; report the mode.
- Upstream rasterization clips negative coordinates in skimage, even without
  an explicit image shape. Our unbounded pixel IoU translates by integer pixels
  before rasterization and uses coordinate sets, preserving translation behavior.
  Agreement tests use valid in-frame rectangles; negative-coordinate differences
  are documented and tested separately.

## 5. Official splits: measured counts, not dataset row estimates

Source binaries: `split/grasp-anything++/train/seen.obj`,
`split/grasp-anything++/test/seen.obj`, `split/grasp-anything++/test/unseen.obj`.
They are pickle binaries, so **line numbers are not meaningful for these files**.
Reproducible evidence is their SHA-256 plus the generated
[`upstream_splits.json`](upstream_splits.json) (line-numbered JSON), produced by
`g2g/inspection/upstream.py:read_split` and `:audit_upstream`.

| File | Entries (objects) | Unique images | Line-numbered count evidence |
|---|---:|---:|---|
| train/seen.obj | 14,516 | 11,616 | `docs/upstream_splits.json:29–30` |
| test/seen.obj | 573 | 420 | `docs/upstream_splits.json:7–8` |
| test/unseen.obj | 230 | 200 | `docs/upstream_splits.json:18–19` |

- Each primitive list entry is `<64-hex-image-id>_<object-id>`, not a part row.
  `grasp_anywhere_data.py:38`, `:43` strips the final `_<part>.pt` for its
  single-digit part assumption, explaining object-level membership.
- No `train/unseen.obj` is present. Train size 15k images is not available
  from the official train list alone: it contains only 11,616 images.
- Object IDs are disjoint across all three lists. Test Seen/Unseen images
  are disjoint, and train/test Seen images are disjoint.
- **37 images overlap train Seen and test Unseen.** Our stricter image-disjoint
  protocol excludes all test images before splitting train/val, leaving at
  most **11,579** eligible train-pool images before val holdout/data checks.
  This changes the official training pool and must be disclosed.
- Union grasps include **only object IDs explicitly permitted by train Seen**,
  not every annotation found in a selected train image. Otherwise other objects
  in the scene may leak supervision outside the intended protocol.
- Counts of part labels/instructions are separate from object split counts;
  actual sampled counts are recorded in `data/manifests/*/protocol.json`.
  The archive audit finds only 8,344 eligible training-pool images with actual
  labels before val holdout, below the upper bound inferred from split IDs alone.
- The loader always reads **test** lists (`grasp_anywhere_data.py:35–43`);
  `seen=True` does not select `train/seen.obj`. Use explicit manifests instead.

## 6. Model audit and implementation boundaries

- GR-ConvNet-3 defaults to 4-channel input (`grconvnet3.py:9`); instantiate RGB
  with 3 channels. Its five residual blocks and transpose-convolution decoder
  can be adapted with license/attribution. Parameter counts must be measured.
- Optional B2 hardcodes 56x56 and 128 fusion channels
  (`inference/models/lgrconvnet3.py:38–45`, `:80`) and detaches vision features
  (`:83`). Do not call this an end-to-end trained visual backbone unchanged.
- Part instruction content, mask shapes/values and actual label tensors were
  subsequently verified: see `REAL_DATA_VERIFICATION.md`. Neither that inspection
  nor CPU checks are evidence of completed GPU R1/R2 experiments.

## 7. Next verification

Confirm the dataset owner's access requirements, inspect ~20 matched real
samples and view ~10 overlays. Record actual repr/shape/dtype, member hashes,
instruction selection and mask matching before implementing a native adapter.
