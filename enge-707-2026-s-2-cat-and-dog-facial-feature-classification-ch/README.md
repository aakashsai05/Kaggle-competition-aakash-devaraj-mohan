# ENGE707 - Cat and Dog Facial Feature Classification

## Project Overview

This individual ENGE707 project develops and evaluates a deep learning pipeline for classifying cat and dog facial features into eight categories:

- CatEye
- CatFace
- CatMouth
- CatNose
- DogEye
- DogFace
- DogMouth
- DogNose

The project uses a pretrained ResNet18 model, duplicate-safe training and validation splitting, and test-time augmentation (TTA).

## Project Structure

| Folder | Description |
|---|---|
| 01_Dataset_Exploration | Dataset statistics, sample images and quality checks |
| 02_Data_Preprocessing | Data quality audit and duplicate-safe splitting |
| 03_Model_Development | ResNet18 training scripts and model configuration |
| 04_Training_Strategy | Training histories, learning curves and analysis |
| 05_Model_Evaluation | Evaluation scripts, confusion matrices and Kaggle predictions |
| 06_Results_Analysis | Model comparisons, error analysis and TTA experiments |

Additional project documentation is available in REPRODUCIBILITY_NOTES.md.

## Dataset Exploration

The project uses the dataset supplied through the ENGE707 Kaggle competition.

- Entries in training labels CSV: 63,660
- Matching locally available training images: 19,921
- Label entries without matching local images: 43,739
- Target classes: 8
- Corrupted available images: 0
- Images with at least one dimension below 32 pixels: 1,336
- Exact duplicate groups identified: 2

Only labels with matching available image files were included in the training experiments.

The available class distribution is recorded in the dataset exploration results.

## Data Preprocessing

Training images were resized to 224 x 224 pixels and normalised using ImageNet channel means and standard deviations.

Training augmentation included:
- Random horizontal flipping
- Random rotations up to 10 degrees

Validation images used deterministic resizing and normalisation.

A duplicate-safe split was created to prevent known exact duplicate groups from crossing between training and validation.

Corrected split:
- Training images: 15,937
- Validation images: 3,984

The corrected split removes identified exact-duplicate overlap but does not establish that all visually similar images are independent.

## Model Development

Architecture: Pretrained ResNet18

The model was adapted to classify eight categories.

Training configuration:
- Loss function: CrossEntropyLoss
- Optimiser: AdamW
- Learning rate: 0.0003
- Batch size: 32
- Training epochs: 5
- Random seed: 42

Most pretrained layers were frozen, while layer4 and the final classification layer were trainable.

## Training Results

### Original ResNet18 Baseline

- Validation accuracy: 98.67%
- Validation macro F1: 98.71%
- Best epoch: 5

The original split contained one known exact duplicate group crossing training and validation.

### Corrected ResNet18

- Validation accuracy: 98.27%
- Validation macro F1: 98.22%
- Validation cross-entropy loss: 0.047968
- Best checkpoint: Epoch 4

The original and corrected models used different validation splits, so their scores are not directly comparable on identical images.

## Test-Time Augmentation

The selected corrected model was evaluated using several test-time augmentation strategies.

| Method | Validation Accuracy | Macro F1 |
|---|---:|---:|
| Horizontal-flip TTA | 0.985944 | 0.985463 |
| Rotation TTA (+/-5 degrees) | 0.985442 | 0.985423 |
| Multi-scale TTA 75-25 | 0.984438 | 0.983866 |
| Multi-scale TTA 50-50 | 0.981677 | 0.980590 |

Horizontal-flip TTA produced the best results among these evaluated inference methods.

This technique combines class probabilities from original and horizontally flipped images without additional model training.

## Kaggle Results

The model generated predictions for 29,443 test images.

| Submission | Public Leaderboard Score |
|---|---:|
| Corrected ResNet18 | 0.9611 |
| Corrected ResNet18 + Horizontal-Flip TTA | 0.9629 |

Best recorded public leaderboard score: 0.9629

The Kaggle evaluation metric should be checked against the official competition scoring description before comparing it directly with local validation metrics.

The final TTA submission is saved as:

05_Model_Evaluation/submission_tta.csv

## Requirements

Tested Python version: 3.12.10

Install the dependencies:

    python -m pip install -r requirements.txt

Check installed package compatibility:

    python -m pip check

## Dataset Setup

Download the authorised competition dataset from Kaggle.

Expected local structure:

    train_labels.csv
    train/train/00001.jpg
    test/test/00001.jpg

The complete image dataset and trained checkpoints are not stored in the GitHub repository.

## Running the Project

Run commands from this competition project directory.

Create the corrected split:

    python 02_Data_Preprocessing/create_clean_split.py

Train the corrected model:

    python 03_Model_Development/train_clean.py --epochs 5 --batch-size 32

Evaluate the corrected model:

    python 05_Model_Evaluation/evaluate_model.py

Calculate validation loss:

    python 05_Model_Evaluation/compute_validation_loss.py

Evaluate horizontal-flip TTA:

    python 05_Model_Evaluation/evaluate_tta.py

Generate the TTA Kaggle submission:

    python 05_Model_Evaluation/create_submission_tta.py

Verify submission CSV files:

    python 05_Model_Evaluation/verify_kaggle_files.py

Training from scratch requires the authorised competition dataset. The pretrained ResNet18 weights may also need to be downloaded on first use.

CPU execution is supported but can be slow.

## Reproducibility and Limitations

- Random seeds were specified during training.
- The corrected validation split is saved for reuse.
- The best checkpoint was selected using validation macro F1.
- Only one corrected training run was reported.
- Training accuracy and per-epoch validation loss were not originally recorded.
- The validation set was also used for model selection.
- Known exact duplicate overlap was removed, but near-duplicate overlap was not fully investigated.
- Differences between Kaggle and validation performance require careful interpretation.

See REPRODUCIBILITY_NOTES.md and 06_Results_Analysis for further details.

## Repository

https://github.com/aakashsai05/Kaggle-competition-aakash-devaraj-mohan
