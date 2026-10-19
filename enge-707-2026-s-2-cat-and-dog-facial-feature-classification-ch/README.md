# ENGE707 - Cat and Dog Facial Feature Classification

## Project Overview
This project develops a deep learning model to classify cat and dog facial features into eight classes.

## Project Structure
- 01_Dataset_Exploration: Dataset statistics, visualisations and quality checks
- 02_Data_Preprocessing: Image preprocessing and augmentation
- 03_Model_Development: ResNet18 model and training code
- 04_Training_Strategy: Training experiments and learning curves
- 05_Model_Evaluation: Model evaluation and Kaggle submissions
- 06_Results_Analysis: Error analysis and model improvements
- 07_Final_Report: Final technical report and supporting evidence

## Dataset
- Available training images: 19,921
- Number of classes: 8
- Corrupted images: 0
- Images with a dimension below 32 pixels: 1,336
- Exact duplicate groups: 2

## Baseline Model
- Architecture: Pretrained ResNet18
- Training images: 15,936
- Validation images: 3,985
- Training epochs: 5
- Best validation accuracy: 98.67%
- Best macro F1-score: 98.71%

## Project Status
- Task 1: Dataset exploration completed
- Task 2: Pending
- Task 3: Baseline model completed
- Task 4: Training analysis pending
- Task 5: Kaggle evaluation pending
- Task 6: Results analysis pending
- Task 7: Final report pending

## Reproducibility
Run the training script from the competition folder:

python 03_Model_Development/train.py --epochs 5 --batch-size 32

Large dataset archives and trained model checkpoints are stored locally and excluded from GitHub.
