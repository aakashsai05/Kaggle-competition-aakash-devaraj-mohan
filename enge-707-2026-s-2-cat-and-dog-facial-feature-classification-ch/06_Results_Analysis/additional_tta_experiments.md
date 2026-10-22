# Additional TTA Experiments

## Objective
Evaluate whether multi-scale or rotation-based test-time
augmentation improves the corrected ResNet18 classifier.

## Validation Results

| Method | Accuracy | Macro F1 |
|---|---:|---:|
| Horizontal Flip TTA | 0.985944 | 0.985463 |
| Multi-scale TTA 50-50 | 0.981677 | 0.980590 |
| Multi-scale TTA 75-25 | 0.984438 | 0.983866 |
| Rotation TTA (+/-5 degrees) | 0.985442 | 0.985423 |

## Findings

Horizontal-flip TTA produced the highest validation accuracy
and macro F1 among the tested methods.

Multi-scale TTA reduced validation performance. The model
was trained using 224x224 inputs, so changing image resolution
may have contributed to the lower scores.

Rotation TTA performed similarly to horizontal-flip TTA,
but did not improve either evaluation metric.

The tested rotations used zero-filled borders on normalised
images, which may have affected the results.

## Decision

Retain horizontal-flip TTA as the preferred inference method.

The best recorded Kaggle public leaderboard score remains
0.9629, achieved using horizontal-flip TTA.

No multi-scale or rotation TTA submissions were made.

## Limitations

These comparisons used one validation split and one trained
checkpoint. Results may differ across datasets or models.
