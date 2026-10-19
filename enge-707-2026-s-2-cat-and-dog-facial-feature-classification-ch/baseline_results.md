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
- First baseline model using transfer learning.
- Training used 19921 available labelled images.
- Additional evaluation and data leakage checks are required.
- Kaggle leaderboard score has not yet been measured.
