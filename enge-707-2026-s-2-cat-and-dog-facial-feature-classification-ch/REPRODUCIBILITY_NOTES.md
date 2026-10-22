# ENGE707 classifier — reproducibility guide

## Repository

GitHub: https://github.com/aakashsai05/Kaggle-competition-aakash-devaraj-mohan

Run commands from the directory containing `train_labels.csv`, `train/`, `test/` and the numbered task folders. This is a **nested competition directory inside the Git repository**, so `Set-Location` to that competition directory before running relative-path scripts.

## Expected local inputs

- `train_labels.csv` with `image_id,class` columns.
- `train/train/00001.jpg` etc., extracted competition training images.
- `test/test/00001.jpg` etc., extracted competition test images.
- `05_Model_Evaluation/sample_submission.csv` containing `image_id,class`.
- `02_Data_Preprocessing/results/clean_train_validation_split.csv` with `image_id,class,split,group_id`.
- `03_Model_Development/checkpoints/best_model_clean.pth` **local** selected checkpoint (ignored by Git).

Do **not** use external datasets. The pretrained ResNet18 weights are standard transfer-learning initialization; the competition training images are the assessment dataset. Initial training may download torchvision pretrained weights.

## Python dependencies

Scripts require Python, PyTorch (`torch`, `torchvision`), NumPy, pandas, Pillow, scikit-learn and Matplotlib. To capture the **exact** versions used on the student's computer:

```powershell
python --version
python -m pip freeze > environment_freeze.txt
```

This guide does not invent package versions. The environment freeze can be reviewed before committing, because it may list unrelated packages.

## Reproduce with the existing selected checkpoint (no retraining)

```powershell
python "05_Model_Evaluation/evaluate_model.py"
python "05_Model_Evaluation/compute_validation_loss.py"
python "05_Model_Evaluation/evaluate_tta.py"
python "05_Model_Evaluation/verify_kaggle_files.py"
```

Expected best selected-checkpoint validation accuracy: **0.9827**; macro F1: **0.9822**. Horizontal-flip TTA validation accuracy: **0.985944**; macro F1: **0.985463**.

## Recreate Kaggle prediction CSVs (runs CPU inference on test data)

```powershell
python "05_Model_Evaluation/create_submission.py"
python "05_Model_Evaluation/create_submission_tta.py"
python "05_Model_Evaluation/verify_kaggle_files.py"
```

Preferred competition submission: `05_Model_Evaluation/submission_tta.csv`. Public score recorded in screenshot: **0.9629**. Source of truth for Kaggle score is the competition page, not the CSV alone.

## More experimental comparisons (optional to rerun)

```powershell
python "05_Model_Evaluation/evaluate_multiscale_tta.py"
python "05_Model_Evaluation/evaluate_rotation_tta.py"
```

Both experiments performed worse than horizontal-flip-only TTA on validation. They were retained as negative results, **not submitted to Kaggle**.

## Retraining (not required to verify the already completed project)

```powershell
python "03_Model_Development/train_clean.py" --epochs 5 --batch-size 32
```

**CAUTION:** This command can overwrite `best_model_clean.pth` and change local results; back up the checkpoint first. Check original scripts and random seeds before claiming bit-for-bit reproducibility. The experiment originally ran on CPU, and small training differences can occur between versions or hardware.

## Evidence and limitations

- Recorded results: `06_Results_Analysis/technical_results_reference.md`.
- Final required items and report planning: `07_Final_Report/submission_and_report_checklist.md`.
- The model checkpoint is not in GitHub because it is large. The submission CSV and source scripts are tracked.
- Corrected dataset split addresses identified **exact** duplicates; it does not prove complete independence from near-duplicates.
- Per-epoch training accuracy/validation loss were not logged, so do not manufacture them.
