"""Generate a Kaggle submission using the best corrected ResNet18 model."""

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = ROOT / "05_Model_Evaluation"

CLASSES = [
    "CatEye", "CatFace", "CatMouth", "CatNose",
    "DogEye", "DogFace", "DogMouth", "DogNose"
]


class TestDataset(Dataset):
    def __init__(self, image_ids, image_dir, transform):
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = int(self.image_ids[index])
        image_path = self.image_dir / f"{image_id:05d}.jpg"

        with Image.open(image_path) as img:
            image = self.transform(img.convert("RGB"))

        return image


def main():
    sample_path = EVAL_DIR / "sample_submission.csv"
    checkpoint_path = (
        ROOT / "03_Model_Development"
        / "checkpoints" / "best_model_clean.pth"
    )
    test_dir = ROOT / "test" / "test"
    output_path = EVAL_DIR / "submission_clean.csv"

    sample = pd.read_csv(sample_path)

    if list(sample.columns) != ["image_id", "class"]:
        raise ValueError("Unexpected sample submission columns")

    if sample.empty or sample["image_id"].isna().any():
        raise ValueError("Missing image IDs in sample submission")

    if sample["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs in sample submission")

    ids = pd.to_numeric(sample["image_id"], errors="raise")

    if ids.isna().any() or not (ids == ids.astype("int64")).all():
        raise ValueError("Image IDs must be integers")

    image_ids = ids.astype("int64").tolist()

    missing = [
        image_id for image_id in image_ids
        if not (test_dir / f"{image_id:05d}.jpg").is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} test images are missing. "
            f"First missing IDs: {missing[:10]}"
        )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    dataset = TestDataset(image_ids, test_dir, transform)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["classes"] != CLASSES:
        raise ValueError("Checkpoint class order does not match")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    predictions = []

    print(f"Device: {device}")
    print(f"Test images: {len(dataset):,}")
    print("Generating Kaggle predictions...", flush=True)

    with torch.inference_mode():
        for batch_number, images in enumerate(loader, start=1):
            outputs = model(images.to(device))
            predicted_ids = outputs.argmax(dim=1)

            predictions.extend(
                predicted_ids.cpu().tolist()
            )

            if batch_number % 100 == 0:
                print(
                    f"Processed {len(predictions):,}/{len(dataset):,} images",
                    flush=True,
                )

    if len(predictions) != len(sample):
        raise RuntimeError("Prediction count does not match sample submission")

    submission = pd.DataFrame({
        "image_id": sample["image_id"],
        "class": [CLASSES[i] for i in predictions],
    })

    submission.to_csv(output_path, index=False)

    print()
    print("Submission created successfully.")
    print(f"Rows: {len(submission):,}")
    print(f"Saved to: {output_path}")

    print("\nFirst five predictions:")
    print(submission.head().to_string(index=False))

    print("\nPredicted class distribution:")
    print(submission["class"].value_counts().to_string())


if __name__ == "__main__":
    main()
