# ENGE707 Assessment 3 — final submission checklist and evidence map

This checklist follows the **adapted / unofficial brief available in this chat**, which itself says the *unmodified official instructions take precedence*. Those copies warn of restrictions on AI-generated report writing. **Write the individual report yourself and verify permitted tool use in the official instructions.** This checklist is logistical, not report content for submission.

## Report sections (target length: 3,000–3,500 words)

| Section | Suggested words | Your evidence to use |
|---|---:|---|
| Introduction/problem definition | 250 | Eight-class prediction, use case, task objective |
| Dataset exploration/data quality | 450 | `01_Dataset_Exploration/results`, class distribution, image-quality and duplicate reports |
| Preprocessing and splitting | 450 | `02_Data_Preprocessing/results`, duplicate-group-safe split, train/val transforms |
| Model architecture and training strategy | 500 | `03_Model_Development/train_clean.py`, `04_Training_Strategy/` |
| Validation evaluation | 450 | `05_Model_Evaluation/results/classification_report.csv`, confusion matrix, selected-checkpoint loss |
| Experiments and Kaggle results | 500 | TTA comparisons, both submissions, leaderboard screenshots |
| Limitations and possible future improvements | 350 | Evidence limitations and unsuccessful ablations |
| Conclusion | 250 | Project-specific lessons and observed outcomes |
| **Total target** | **3,200** | **Write in your own words, then verify count** |

These section names and word budgets are planning aids only. The report should use original first-person interpretation and must follow the official instructions.

## Supporting visual evidence

- [ ] Representative images for each of eight classes.
- [ ] Image-class frequency graph.
- [ ] Image quality / duplicate investigation summary.
- [ ] Corrected-training and validation-performance curves. Existing recorded plots are for the original baseline: label them accurately; do not present original baseline curves as corrected-training curves.
- [ ] Corrected-model confusion matrix figure and class-level precision/recall/F1 table.
- [ ] TTA comparison table (standard, horizontal flip, multi-scale, rotation).
- [ ] Kaggle leaderboard screenshots showing score 0.9611 and 0.9629.
- [ ] GitHub commit-history evidence and link.

## Reproducibility / code deliverables

- [ ] `train_clean.py` and the duplicate-safe split are available.
- [ ] Model checkpoint exists **locally**; `.pth` and data ZIPs remain excluded from GitHub unless specifically required by course rules.
- [ ] `evaluate_model.py` reproduces accuracy 0.9827 and macro F1 0.9822.
- [ ] `evaluate_tta.py` reproduces accuracy 0.985944 and macro F1 0.985463.
- [ ] `verify_kaggle_files.py` passes for 29,443 prediction rows.
- [ ] `compute_validation_loss.py` has been run; save the resulting JSON and interpret it as one-checkpoint loss.
- [ ] Dependencies / Python and PyTorch versions recorded (`python --version` and `python -m pip freeze`); include complete environment versions if needed.
- [ ] Git status clean and pushed; any pending Commit 8 experimental files included appropriately.
- [ ] README explains entry points, paths, how to reproduce key tests, and known environment limitations.

## Administrative requirements — confirm against official assessment brief

- [ ] Competition account / Kaggle username recorded.
- [ ] Final submission `submission_tta.csv` and Kaggle evidence submitted where requested.
- [ ] Individual report within the required word count, with sources/references and original writing.
- [ ] Similarity report / Turnitin requirement satisfied (stated threshold ≤20% in adapted brief).
- [ ] At least two meaningful GitHub commits per week evidenced; weekly history verified.
- [ ] Demonstration/viva preparation completed (show how dataset, split, architecture, checkpoint selection, TTA and Kaggle file generation work).
- [ ] Dates, submission links, formats, and permitted AI use verified from official AUT course materials.

## Caveats to address in your own analysis

1. **63,660** CSV labels exist, but only **19,921** matching training JPGs were available locally. Explain which data was actually used.
2. The corrected split only prevents overlap of identified exact-duplicate groups, not all near-duplicates.
3. Baseline and corrected-model validation splits differ; a score difference is not causal evidence that duplicate removal reduced performance.
4. The selected corrected checkpoint is **Epoch 4**; Epoch 5 validation worsened even as training loss declined.
5. Per-epoch validation loss was never logged, and a retrospective per-epoch curve should not be invented.
6. Kaggle's **0.9629 public score** is separate from local 0.985463 validation macro F1. Verify metric before interpreting the numerical gap.
7. No multi-scale/rotation versions were submitted to Kaggle; only their validation comparisons were observed.
