"""Check both saved Kaggle submission CSVs against the supplied sample schema.
Run from the competition project root:
    python 05_Model_Evaluation/verify_kaggle_files.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "05_Model_Evaluation"
CLASSES = {"CatEye", "CatFace", "CatMouth", "CatNose", "DogEye", "DogFace", "DogMouth", "DogNose"}


def main():
    sample = pd.read_csv(EVAL / "sample_submission.csv")
    if list(sample.columns) != ["image_id", "class"]:
        raise ValueError("Sample submission has unexpected columns")
    if sample["image_id"].duplicated().any():
        raise ValueError("Sample has duplicate IDs")
    for filename in ("submission_clean.csv", "submission_tta.csv"):
        path = EVAL / filename
        if not path.exists():
            raise FileNotFoundError(path)
        submission = pd.read_csv(path)
        assert list(submission.columns) == ["image_id", "class"], filename
        assert len(submission) == len(sample) == 29443, filename
        assert submission["image_id"].equals(sample["image_id"]), filename
        assert submission["class"].notna().all(), filename
        assert set(submission["class"]).issubset(CLASSES), filename
        print(f"PASS: {filename} ({len(submission):,} predictions)")
    print("Note: CSV validation does not establish hidden Kaggle test-set accuracy.")


if __name__ == "__main__":
    main()
