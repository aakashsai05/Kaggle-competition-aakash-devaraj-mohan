# ResNet18 Baseline Results

Model: ResNet18 (pretrained)
Classification task: 8 classes
Training images: 15936
Validation images: 3985
Training device: CPU
Epochs: 5

| Epoch | Training Loss | Validation Accuracy | Macro F1 |
|-------|---------------|---------------------|----------|
| 1 | 0.2095 | 0.9731 | 0.9722 |
| 2 | 0.0726 | 0.9757 | 0.9760 |
| 3 | 0.0507 | 0.9749 | 0.9757 |
| 4 | 0.0418 | 0.9854 | 0.9853 |
| 5 | 0.0353 | 0.9867 | 0.9871 |

Best model: Epoch 5
Best validation accuracy: 98.67%
Best macro F1: 98.71%
Notes:
- This was the initial ResNet18 baseline using transfer learning.
- The original validation split contained known exact duplicate overlap.
- A corrected duplicate-safe split was subsequently created.
- The corrected model achieved 98.27% validation accuracy and 98.22% macro F1.
- Horizontal-flip TTA improved the corrected model's validation accuracy to 98.59%.
- The best recorded Kaggle public leaderboard score was 0.9629.
- The original and corrected validation results are not directly comparable because they used different validation splits.

