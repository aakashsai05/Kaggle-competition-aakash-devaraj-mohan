"""Generate Kaggle predictions using horizontal-flip TTA."""

from pathlib import Path
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

from create_submission import CLASSES, TestDataset

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "05_Model_Evaluation"

def main():
    sample = pd.read_csv(EVAL_DIR / "sample_submission.csv")
    test_dir = ROOT / "test" / "test"
    checkpoint_path = ROOT / "03_Model_Development" / "checkpoints" / "best_model_clean.pth"
    output_path = EVAL_DIR / "submission_tta.csv"

    if list(sample.columns) != ["image_id", "class"]:
        raise ValueError("Incorrect submission columns")
    if sample["image_id"].isna().any() or sample["image_id"].duplicated().any():
        raise ValueError("Missing or duplicate image IDs")

    ids = pd.to_numeric(sample["image_id"], errors="raise")
    if not (ids == ids.astype("int64")).all():
        raise ValueError("Image IDs must be integers")
    image_ids = ids.astype("int64").tolist()

    missing = [
        i for i in image_ids
        if not (test_dir / f"{i:05d}.jpg").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Missing test images: {missing[:10]}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])

    dataset = TestDataset(image_ids, test_dir, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if checkpoint["classes"] != CLASSES:
        raise ValueError("Checkpoint class order mismatch")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    predictions = []

    print(f"Device: {device}", flush=True)
    print(f"Test images: {len(dataset):,}", flush=True)
    print("Generating TTA predictions...", flush=True)

    with torch.inference_mode():
        for batch_number, images in enumerate(loader, 1):
            images = images.to(device)

            original = torch.softmax(model(images), dim=1)
            flipped = torch.softmax(
                model(torch.flip(images, dims=[3])), dim=1
            )

            averaged = (original + flipped) / 2
            predictions.extend(averaged.argmax(dim=1).cpu().tolist())

            if batch_number % 100 == 0:
                print(
                    f"Processed {len(predictions):,}/{len(dataset):,}",
                    flush=True
                )

    if len(predictions) != len(sample):
        raise RuntimeError("Prediction count mismatch")

    submission = pd.DataFrame({
        "image_id": sample["image_id"],
        "class": [CLASSES[i] for i in predictions]
    })
    submission.to_csv(output_path, index=False)

    saved = pd.read_csv(output_path)
    assert list(saved.columns) == ["image_id", "class"]
    assert saved["image_id"].equals(sample["image_id"])
    assert len(saved) == len(sample)
    assert saved["class"].notna().all()
    assert set(saved["class"]).issubset(set(CLASSES))

    print("\nTTA SUBMISSION CREATED SUCCESSFULLY")
    print(f"Rows: {len(saved):,}")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
