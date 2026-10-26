# Real-data verification — 2026-10-26

The user confirmed completion of the owner's download form/license in this
session before sample extraction. HF API/HEAD checks found both repositories
public (`gated=false`, HTTP 200); a token was not needed for these checks.

Pinned dataset revisions are recorded in `docs/dataset_access.json` and
`g2g/inspection/samples.py`. Extraction checks ZIP member CRC and records SHA-256.

## Observed before implementing the native parser

- Labels extracted from `grasp_label_positive/<image>_<object>_<part>.pt`
  deserialize with `torch.load(weights_only=True)` as **float32 tensors N×6**.
  Two inspected examples have shapes 13×6 and 10×6. Conversion of the last
  five fields follows the pinned upstream code; the first field is preserved
  in source data but not assigned an unverified physical meaning.
- `grasp_instructions/<same-stem>.pkl` is a **plain string**, e.g.
  `Lift plant by its leaves.` and `Grip scissors on its cutting blades.`
  This validates using the instruction file, not upstream object `queries[obj_id]`.
- `part_mask/<same-stem>.npy` is **416×416 uint8 with values 0 and 1** in the
  inspected examples. Twenty matched masks have been extracted.
- Part 0 is **not guaranteed to exist** for every official object ID. The
  VERIFY sampler checks inventory existence and records skipped missing members.
  Benchmark manifests must enumerate actual labels rather than invent `_0` rows.
- Archive counts (including the directory entry) for positive labels,
  instructions and masks are each 4,412,384. This is **not** a split row count
  and does not itself prove that all filenames match across archives.

## Gate result

Twenty matched images are 416×416 RGB. All 20 label/instruction/mask/image sets
parsed successfully; 10 overlays were rendered and the full contact sheet was
visually inspected. Evidence: `real_file_inspection.json`,
`verification_selection.json`, `overlay_verification.json`, and
`figs/schema_contact_sheet.jpg`. The mapping and geometry gate passes.

**Annotation quality caveat:** some references are semantically noisy (e.g.
an instruction naming a ball over a lime), some part masks/grasps disagree,
and one book sample has no grasp centres inside its mask. These are recorded,
not silently corrected or removed. A mask oracle is not necessarily perfect
under grasp GT. This limits interpreting the planned error decomposition.

The VERIFY sample set intentionally selects existing part-0 labels across
distinct eligible train images; it is not an unbiased benchmark subset and
does not contain complete union targets. Training manifests will enumerate
every actual part label for permitted objects instead.

Object-level masks need no additional download because that training ablation
is optional and currently out of scope.
