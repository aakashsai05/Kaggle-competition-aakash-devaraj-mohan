"""Evaluate small-angle rotation TTA against horizontal-flip TTA."""

import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "03_Model_Development"))
from train_clean import CLASSES, FacialFeaturesDataset


def main():
    split = pd.read_csv(
        ROOT / "02_Data_Preprocessing" / "results"
        / "clean_train_validation_split.csv"
    )

    validation = split[
        split["split"] == "validation"
    ].reset_index(drop=True)

    if len(validation) != 3984:
        raise ValueError("Unexpected validation count")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        ),
    ])

    dataset = FacialFeaturesDataset(
        validation, ROOT / "train" / "train", transform
    )

    loader = DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=0
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        ROOT / "03_Model_Development" / "checkpoints"
        / "best_model_clean.pth",
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["classes"] != CLASSES:
        raise ValueError("Checkpoint class order mismatch")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()

    actual = []
    flip_predictions = []
    rotation_predictions = []

    print(f"Device: {device}", flush=True)
    print("Testing rotation TTA...", flush=True)

    with torch.inference_mode():
        for batch_number, (images, labels) in enumerate(loader, 1):
            images = images.to(device)

            p_original = torch.softmax(model(images), dim=1)

            p_flip = torch.softmax(
                model(torch.flip(images, dims=[3])), dim=1
            )

            rotated_plus = TF.rotate(
                images, angle=5,
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )

            rotated_minus = TF.rotate(
                images, angle=-5,
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )

            p_plus = torch.softmax(model(rotated_plus), dim=1)
            p_minus = torch.softmax(model(rotated_minus), dim=1)

            flip_probs = (p_original + p_flip) / 2

            rotation_probs = (
                p_original + p_flip + p_plus + p_minus
            ) / 4

            actual.extend(labels.tolist())

            flip_predictions.extend(
                flip_probs.argmax(1).cpu().tolist()
            )

            rotation_predictions.extend(
                rotation_probs.argmax(1).cpu().tolist()
            )

            if batch_number % 40 == 0:
                print(
                    f"Processed {len(actual):,}/3,984",
                    flush=True
                )

    results = pd.DataFrame([
        {
            "method": "Horizontal Flip TTA",
            "accuracy": accuracy_score(actual, flip_predictions),
            "macro_f1": f1_score(
                actual, flip_predictions,
                average="macro", zero_division=0
            ),
        },
        {
            "method": "Rotation TTA (+/-5 degrees)",
            "accuracy": accuracy_score(actual, rotation_predictions),
            "macro_f1": f1_score(
                actual, rotation_predictions,
                average="macro", zero_division=0
            ),
        },
    ])

    if abs(results.iloc[0]["macro_f1"] - 0.985463) > 0.0001:
        raise RuntimeError("Baseline TTA result does not match")

    output = (
        ROOT / "05_Model_Evaluation" / "results"
        / "rotation_tta_comparison.csv"
    )

    results.to_csv(output, index=False)

    print("\nROTATION TTA RESULTS")
    print(results.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
