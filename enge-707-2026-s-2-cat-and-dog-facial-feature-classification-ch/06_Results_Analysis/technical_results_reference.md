# ENGE707 technical results reference — checked against project terminal outputs

**Purpose:** factual source-of-truth notes for reproducing and reviewing experiments. Not a submission-ready report. All metrics listed here come from recorded terminal outputs / local evaluation shown during development. Round-off differences may occur.

## Dataset and preprocessing

| Item | Observed value |
|---|---:|
| Labels in CSV | 63,660 |
| Labels with corresponding local JPG | 19,921 |
| Labels lacking local JPG | 43,739 |
| Kaggle test images processed | 29,443 |
| Image classes | 8 |
| Corrupt local images detected | 0 |
| Exact duplicate image groups identified | 2 |
| Exact duplicate groups crossing original validation split | 1 |
| Exact duplicate groups crossing corrected split | 0 |
| Corrected training images | 15,937 |
| Corrected validation images | 3,984 |
| Images with a side shorter than 32 pixels | 1,336 |
| Distinct image sizes | 3,055 |

Class frequencies in the **19,921 locally available training images**, not the 63,660-row CSV:

| Class | Images |
|---|---:|
| CatEye | 4,347 |
| DogEye | 3,665 |
| CatMouth | 2,201 |
| CatFace | 2,169 |
| CatNose | 2,142 |
| DogNose | 1,837 |
| DogMouth | 1,789 |
| DogFace | 1,771 |

Caution: the corrected split prevents crossing of the **known exact-duplicate groups**. This alone does not establish the absence of near-duplicates or related facial crops across the split.

## Model configuration — `train_clean.py`

- Image size: **224 × 224 RGB**.
- Labels in checkpoint order: CatEye, CatFace, CatMouth, CatNose, DogEye, DogFace, DogMouth, DogNose.
- Backbone: torchvision pretrained **ResNet18**; freeze early layers, train `layer4` and newly replaced eight-output `fc` classifier.
- Optimizer: **AdamW**, learning rate **0.0003**; batch size **32**; **5 epochs**; seed **42**.
- Loss: **cross-entropy**. Validation selected by **macro F1**.
- Training transforms: resize 224, random horizontal flip, random rotation ±10°, tensor conversion, ImageNet mean/std normalization. Validation/test: resize 224, tensor, same normalization (no random augmentations).
- Device used for runs: CPU.
- Checkpoint: `03_Model_Development/checkpoints/best_model_clean.pth` (saved locally and intentionally excluded from GitHub).

## Training history

| Epoch | Corrected training loss | Corrected val. accuracy | Corrected val. macro F1 |
|---|---:|---:|---:|
| 1 | 0.2082 | 0.9739 | 0.9726 |
| 2 | 0.0750 | 0.9792 | 0.9792 |
| 3 | 0.0531 | 0.9802 | 0.9798 |
| **4 (selected)** | **0.0424** | **0.9827** | **0.9822** |
| 5 | 0.0330 | 0.9674 | 0.9691 |

Original baseline best (different split): validation accuracy **0.9867**, macro F1 **0.9871**, at epoch 5. Comparison across different validation splits is *not* a controlled estimate of the effect of removing exact duplicates.

The Epoch 5 validation decline despite lower training loss is consistent with possible overfitting or other training variability, but not definitive evidence of its cause. **Per-epoch training accuracy and validation loss were not recorded.** `compute_validation_loss.py` can calculate a *single* validation loss for the selected checkpoint, not a five-epoch validation-loss curve.

## Corrected validation confusion matrix — 3,984 images

Rows = actual; columns = predicted. The full matrix is saved in `05_Model_Evaluation/results/confusion_matrix.csv`.

- Correct: **3,915**; incorrect: **69**.
- Accuracy: **0.9827**; macro F1: **0.9822**.
- Frequent off-diagonal errors: CatMouth→CatNose **14**, CatEye→DogEye **10**, DogNose→DogEye **7**, DogNose→CatNose **5**.
- CatFace: **431/434** correct; DogFace: **353/354** correct.
- CatNose precision is ~**0.94**, despite recall ~**0.99**: many other-class inputs predicted as CatNose.
- CatMouth recall ~**0.96** and DogNose recall ~**0.96**.

Do not claim the visual cause of these errors without inspecting examples. Candidate explanations (crop overlap, appearance similarity, scale, blur) are hypotheses.

## TTA experiments — **same corrected validation split and selected checkpoint**

| Method | Validation accuracy | Macro F1 |
|---|---:|---:|
| Standard inference | 0.9827 | 0.9822 |
| **Horizontal-flip TTA** | **0.985944** | **0.985463** |
| Multi-scale, 224/256 50:50 | 0.981677 | 0.980590 |
| Multi-scale, 224/256 75:25 | 0.984438 | 0.983866 |
| Rotation TTA ±5° + flip | 0.985442 | 0.985423 |

Horizontal-flip TTA averages softmax probabilities from original and horizontally flipped inputs. Among these tested settings it performed best on this single validation split. The ±5° rotation trial used zero fill after normalization and should not be overgeneralised.

## Kaggle records (from user-provided screenshots)

| Submission | Public leaderboard score |
|---|---:|
| Corrected ResNet18 | 0.9611 |
| **Corrected ResNet18 + horizontal flip TTA** | **0.9629** |

The displayed leaderboard place of **20th** applied to the first submission screenshot and may have changed. Scores are recorded as Kaggle *public score*, not automatically called accuracy or F1 until the competition metric is independently confirmed. Hidden-test score and final placement are not yet known.

The final selected submission file is `05_Model_Evaluation/submission_tta.csv`, with `image_id,class` header and **29,443** predictions. Validate it using `verify_kaggle_files.py`.

## Open evidence gaps

- Capture the **Kaggle username** and screenshots with both public scores, plus final selection status if relevant.
- Verify the actual **competition scoring metric** before comparing it numerically to accuracy/F1.
- Run `compute_validation_loss.py` for the **selected checkpoint's validation loss**; per-epoch loss history cannot be retrospectively reconstructed from this checkpoint.
- Check whether test images or copied labels are excluded from GitHub and `.gitignore` works.
- Add sample images, confusion-matrix figure, and corrected learning-curve figure to the final evidence bundle (without altering original data).
- Confirm the report submission format, official AI policy, due date, Turnitin rule, and demonstration/viva requirements against the **unmodified official brief**.
