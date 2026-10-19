# Task 4 - Training Strategy and Analysis

## 1. Model Configuration

The baseline classifier uses a pretrained ResNet18 architecture implemented in PyTorch.

Training configuration:
- Architecture: ResNet18
- Input dimensions: 224 x 224 pixels
- Output classes: 8
- Optimiser: AdamW
- Learning rate: 0.0003
- Loss function: CrossEntropyLoss
- Batch size: 32
- Training epochs: 5
- Random seed: 42
- Training device: CPU

The pretrained network was adapted by replacing the final fully connected layer with an eight-class output layer.

Most pretrained parameters were frozen, while layer4 and the new classification layer were trained.

## 2. Training and Validation Strategy

The original baseline used 19,921 available labelled images.

Training images: 15,936
Validation images: 3,985

An 80/20 stratified split was used to maintain approximately equal class proportions across the two sets.

Training augmentation included:
- Resizing to 224 x 224 pixels
- Random horizontal flipping
- Random rotations up to 10 degrees
- Tensor conversion
- ImageNet normalisation

Validation images were resized and normalised without random augmentation.

## 3. Training Results

| Epoch | Training Loss | Validation Accuracy | Macro F1 |
|---|---|---|---|
| 1 | 0.2095 | 97.31% | 97.22% |
| 2 | 0.0726 | 97.57% | 97.60% |
| 3 | 0.0507 | 97.49% | 97.57% |
| 4 | 0.0418 | 98.54% | 98.53% |
| 5 | 0.0353 | 98.67% | 98.71% |

The highest recorded validation accuracy and macro F1-score occurred in Epoch 5.

## 4. Training Behaviour

Training loss decreased consistently from 0.2095 to 0.0353 across five epochs.

Validation accuracy improved overall from 97.31% to 98.67%.

A small validation accuracy decrease occurred between Epochs 2 and 3, followed by improvements in Epochs 4 and 5.

These results indicate that the model learned effectively during the baseline experiment.

However, training accuracy and validation loss were not recorded. Therefore, overfitting and underfitting cannot be evaluated comprehensively using these results alone.

## 5. Generalisation Strategies

Transfer learning reduced the amount of model training required.

Random image augmentation introduced variation in the training images.

Freezing earlier pretrained layers limited the number of trainable parameters.

Model checkpoints were selected using validation macro F1-score.

## 6. Data Leakage Limitation

A subsequent preprocessing audit identified one exact duplicate image group crossing the original training and validation sets.

A corrected duplicate-safe split was created during Task 2.

Therefore, the reported 98.67% accuracy should be treated as the original baseline validation result, not a confirmed leakage-free result.

The corrected split has not yet been used to train and evaluate a new model.

Exact duplicate detection also does not rule out visually similar images or related crops crossing the split.

## 7. Future Improvements

Further evaluation should include:
- Training and validation loss curves
- Training and validation accuracy curves
- Evaluation using the duplicate-safe split
- Per-class precision, recall and F1-score
- Confusion matrix and misclassification analysis
- Comparison with an alternative model or training configuration

## 8. Conclusion

The pretrained ResNet18 baseline achieved high local validation performance after five epochs.

The results demonstrate that transfer learning is a promising approach for this eight-class classification problem.

However, corrected validation experiments and additional evaluation are needed before drawing stronger conclusions about generalisation to unseen Kaggle test images.
