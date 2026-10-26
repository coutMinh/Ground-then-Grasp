# Third-party sources

## LGD / GR-ConvNet reference

- Repository: <https://github.com/Fsoft-AIC/LGD>
- Audited commit: `501904dc5c30c8fad58d90f7e4f49b4095f11eb0`
- License: MIT, Copyright (c) 2024 An Vuong.
- Source clone lives at sibling `../third_party/LGD/`, outside this repository.
- `docs/DATA_SCHEMA.md` records file:line references at this exact commit.
- Any adapted source file includes attribution and a copy of the upstream
  license is retained in `licenses/LGD-MIT.txt` when code is vendored.

## Models and dataset

- CLIPSeg pretrained weights: `CIDAS/clipseg-rd64-refined` (optional dependency).
  Download occurs only when explicitly constructing the pretrained grounder.
- Grasp-Anything and Grasp-Anything++ are external datasets. Follow the owner's
  access/license requirements. They are not redistributed in this repository.
