ENGE707 - Test-Time Augmentation Experiment

Model: Corrected ResNet18
Checkpoint: best_model_clean.pth
Selected training epoch: 4

Method:
Horizontal-flip test-time augmentation.
Predictions from the original and horizontally flipped
images were combined by averaging class probabilities.

Validation Results:
Standard accuracy: 0.9827
TTA accuracy: 0.9859
Standard macro F1: 0.9822
TTA macro F1: 0.9855

Kaggle Results:
Standard public score: 0.9611
TTA public score: 0.9629
Score improvement: +0.0018

The TTA experiment improved both validation performance
and the Kaggle public leaderboard score.

The same trained checkpoint was used for both submissions.
No additional model training was required.
