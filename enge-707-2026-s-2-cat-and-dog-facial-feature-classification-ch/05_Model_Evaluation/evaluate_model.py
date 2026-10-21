"""Task 5: Evaluate the best duplicate-safe ResNet18 checkpoint."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "03_Model_Development"))

from train_clean import CLASSES, FacialFeaturesDataset


def main():
    checkpoint_path = (
        ROOT / "03_Model_Development" / "checkpoints" / "best_model_clean.pth"
    )
    split_path = (
        ROOT / "02_Data_Preprocessing" / "results"
        / "clean_train_validation_split.csv"
    )
    image_dir = ROOT / "train" / "train"
    output_dir = ROOT / "05_Model_Evaluation" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    if not split_path.is_file():
        raise FileNotFoundError(split_path)

    split_df = pd.read_csv(split_path)

    if set(split_df["split"]) != {"train", "validation"}:
        raise ValueError("Unexpected split assignments")
    if split_df["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs in the split")
    if set(split_df.loc[split_df["split"] == "train", "group_id"]) & set(
        split_df.loc[split_df["split"] == "validation", "group_id"]
    ):
        raise ValueError("Duplicate groups overlap between splits")

    val_df = split_df[split_df["split"] == "validation"].copy()
    val_df = val_df.reset_index(drop=True)

    if len(val_df) != 3984:
        raise ValueError(f"Unexpected validation count: {len(val_df)}")
    if val_df["class"].isna().any() or not set(val_df["class"]).issubset(CLASSES):
        raise ValueError("Unexpected validation class labels")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    dataset = FacialFeaturesDataset(val_df, image_dir, transform)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reconstruct the exact ResNet18 architecture used during training.
    # No pretrained download is needed because we load saved weights.
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["classes"] != CLASSES:
        raise ValueError("Checkpoint classes do not match training classes")

    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    actual = []
    predicted = []

    print(f"Device: {device}")
    print(f"Validation images: {len(dataset):,}")
    print("Evaluating saved corrected ResNet18 model...")

    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images.to(device))
            predictions = outputs.argmax(dim=1)

            actual.extend(labels.tolist())
            predicted.extend(predictions.cpu().tolist())

    accuracy = accuracy_score(actual, predicted)
    macro_f1 = f1_score(
        actual, predicted, average="macro", zero_division=0
    )

    label_ids = list(range(len(CLASSES)))

    report_text = classification_report(
        actual,
        predicted,
        labels=label_ids,
        target_names=CLASSES,
        zero_division=0,
    )

    report_dict = classification_report(
        actual,
        predicted,
        labels=label_ids,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report_dict).transpose().to_csv(
        output_dir / "classification_report.csv",
        index_label="class",
    )

    matrix = confusion_matrix(
        actual, predicted, labels=label_ids
    )

    pd.DataFrame(
        matrix,
        index=CLASSES,
        columns=CLASSES,
    ).to_csv(output_dir / "confusion_matrix.csv")

    predictions_df = pd.DataFrame({
        "image_id": val_df["image_id"].values,
        "actual_class": [CLASSES[i] for i in actual],
        "predicted_class": [CLASSES[i] for i in predicted],
    })
    predictions_df.to_csv(
        output_dir / "validation_predictions.csv",
        index=False,
    )

    # Create a readable confusion matrix image.
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Corrected ResNet18 - Validation Confusion Matrix")

    threshold = matrix.max() / 2

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=9,
            )

    fig.tight_layout()
    fig.savefig(
        output_dir / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    summary = (
        "Corrected ResNet18 - Model Evaluation\n"
        "--------------------------------------\n"
        f"Validation images: {len(dataset)}\n"
        f"Validation accuracy: {accuracy:.4f}\n"
        f"Validation macro F1: {macro_f1:.4f}\n"
        "Checkpoint: best_model_clean.pth\n"
        "Selected using best validation macro F1\n\n"
        "Classification Report:\n"
        f"{report_text}\n"
    )

    (output_dir / "evaluation_summary.txt").write_text(
        summary, encoding="utf-8"
    )

    print()
    print(summary)
    print(f"Evaluation files saved in: {output_dir}")


if __name__ == "__main__":
    main()
