# Blockers

Real-file schema verification and CPU data checks have passed. They are not
evidence of completed GPU experiments or final benchmark results.

| ID | Condition | Affected gate | Fallback / resolution |
|---|---|---|---|
| B01 | Local implementation not yet published to the supplied GitHub repo | molab clone of project | Remote configured as `https://github.com/coutMinh/Ground-then-Grasp.git`; publish before running notebook 00. |
| B02 (resolved) | Dataset terms/access initially unconfirmed | F2 | User confirmed form/license; public HTTP access checked; 20 real matched samples verified and 49 smoke rows validated. |
| B03 | No molab session available to this local agent | GPU probe/training | User runs external notebook; save probe and artifacts outside checkout. |
| B04 | Remote persistence/session limits unverified | Long jobs | Probe records unknown, checkpoints stored externally; user verifies persistence. |
| B05 (data caveat) | Annotation/mask inconsistencies observed in real samples | Interpretation of grounding/oracles | Preserve raw data; report diagnostics, do not silently clean test samples or assume oracle-mask success is perfect. |
| B06 (scope limit) | Only 8,344 eligible train-pool images have labels in the pinned release | Original 15k-image training plan | Choose a measured subset below this limit, allowing val holdout; exact counts/hashes live in manifest protocol JSON. |
| B07 (open gate) | Reduced CPU prior reaches 0.5 train-union success after 200 steps on 8 images | Full-capacity overfit acceptance | Run the supplied GPU overfit configs on the 32-image manifest before launching full R1/R2; inspect predictions if success/mIoU remains low. |
| B08 (resolved) | Transient Windows checkpoint replace and CSV-lock races | CPU run stability | Bounded atomic-replace retry, preserved failed run, successful resume to 200 steps, OS-backed registry locks and regression tests. |
| B09 (cloud environment) | Molab CLIPSeg import fails with `operator torchvision::nms does not exist`; traceback loads torch from `/tmp/uv-venv` and torchvision from `/usr/local` | Cloud CLIPSeg gate | Notebook 00 now installs the official torch 2.7.1 / torchvision 0.22.1 cu128 pair and explicitly reinstalls the vision wheel in its interpreter when CUDA setup is selected. Full pytest output is saved. Cloud retest is still required; local tests do not resolve this gate. |
