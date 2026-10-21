"""Compare standard ResNet18 predictions against horizontal-flip TTA."""

import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "03_Model_Development"))

from train_clean import CLASSES, FacialFeaturesDataset


def main():
    checkpoint_path = (
        ROOT / "03_Model_Development" / "checkpoints"
        / "best_model_clean.pth"
    )
    split_path = (
        ROOT / "02_Data_Preprocessing" / "results"
        / "clean_train_validation_split.csv"
    )
    output_dir = ROOT / "05_Model_Evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    split_df = pd.read_csv(split_path)
    val_df = split_df[
        split_df["split"] == "validation"
    ].reset_index(drop=True)

    if len(val_df) != 3984:
        raise ValueError("Unexpected validation image count")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    dataset = FacialFeaturesDataset(
        val_df, ROOT / "train" / "train", transform
    )

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
        raise ValueError("Checkpoint class order mismatch")

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    actual = []
    baseline_predictions = []
    tta_predictions = []

    print(f"Device: {device}", flush=True)
    print(f"Validation images: {len(dataset):,}", flush=True)
    print("Comparing baseline and horizontal-flip TTA...", flush=True)

    with torch.inference_mode():
        for batch_number, (images, labels) in enumerate(
            loader, start=1
        ):
            images = images.to(device)

            original_logits = model(images)
            flipped_logits = model(
                torch.flip(images, dims=[3])
            )

            original_probs = torch.softmax(
                original_logits, dim=1
            )
            flipped_probs = torch.softmax(
                flipped_logits, dim=1
            )

            avg_probs = (original_probs + flipped_probs) / 2

            baseline_predictions.extend(
                original_logits.argmax(dim=1).cpu().tolist()
            )
            tta_predictions.extend(
                avg_probs.argmax(dim=1).cpu().tolist()
            )
            actual.extend(labels.tolist())

            if batch_number % 40 == 0:
                print(
                    f"Processed {len(actual):,}/{len(dataset):,}",
                    flush=True,
                )

    baseline_acc = accuracy_score(actual, baseline_predictions)
    baseline_f1 = f1_score(
        actual, baseline_predictions,
        average="macro", zero_division=0
    )

    tta_acc = accuracy_score(actual, tta_predictions)
    tta_f1 = f1_score(
        actual, tta_predictions,
        average="macro", zero_division=0
    )

    if abs(baseline_acc - 0.9827) > 0.0001:
        raise RuntimeError(
            "Baseline accuracy does not match saved evaluation. "
            "Check the checkpoint, split and transforms."
        )

    comparison = pd.DataFrame([
        {
            "method": "Standard",
            "accuracy": baseline_acc,
            "macro_f1": baseline_f1,
            "correct": sum(
                a == p for a, p in zip(actual, baseline_predictions)
            ),
        },
        {
            "method": "Horizontal Flip TTA",
            "accuracy": tta_acc,
            "macro_f1": tta_f1,
            "correct": sum(
                a == p for a, p in zip(actual, tta_predictions)
            ),
        },
    ])

    comparison.to_csv(
        output_dir / "tta_comparison.csv", index=False
    )

    pd.DataFrame({
        "image_id": val_df["image_id"].values,
        "actual_class": [CLASSES[i] for i in actual],
        "standard_prediction": [
            CLASSES[i] for i in baseline_predictions
        ],
        "tta_prediction": [
            CLASSES[i] for i in tta_predictions
        ],
    }).to_csv(
        output_dir / "tta_validation_predictions.csv",
        index=False,
    )

    print("\nVALIDATION RESULTS")
    print("------------------------------")
    print(f"Standard accuracy: {baseline_acc:.4f}")
    print(f"Standard macro F1: {baseline_f1:.4f}")
    print(f"TTA accuracy:      {tta_acc:.4f}")
    print(f"TTA macro F1:      {tta_f1:.4f}")
    print("------------------------------")
    print(f"Accuracy change: {tta_acc - baseline_acc:+.4f}")
    print(f"Macro F1 change: {tta_f1 - baseline_f1:+.4f}")

    if tta_f1 > baseline_f1:
        print("\nTTA improved validation macro F1.")
    elif tta_f1 < baseline_f1:
        print("\nTTA reduced validation macro F1.")
    else:
        print("\nTTA matched validation macro F1.")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
