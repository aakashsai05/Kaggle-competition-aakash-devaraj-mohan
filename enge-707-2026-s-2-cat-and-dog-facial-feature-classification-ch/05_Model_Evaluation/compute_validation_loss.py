from __future__ import annotations

import json
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
from train_clean import CLASSES, FacialFeaturesDataset  # noqa: E402


def main() -> None:
    checkpoint_path = ROOT / "03_Model_Development" / "checkpoints" / "best_model_clean.pth"
    split_path = ROOT / "02_Data_Preprocessing" / "results" / "clean_train_validation_split.csv"
    image_dir = ROOT / "train" / "train"
    out_path = ROOT / "05_Model_Evaluation" / "results" / "selected_checkpoint_validation_metrics.json"

    for required in (checkpoint_path, split_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)

    frame = pd.read_csv(split_path)
    required_columns = {"image_id", "class", "split", "group_id"}
    if not required_columns.issubset(frame.columns):
        raise ValueError(f"Split missing: {required_columns - set(frame.columns)}")
    if frame["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs in split")
    if set(frame["split"]) != {"train", "validation"}:
        raise ValueError("Unexpected split labels")
    train_groups = set(frame.loc[frame["split"] == "train", "group_id"])
    val_groups = set(frame.loc[frame["split"] == "validation", "group_id"])
    if train_groups & val_groups:
        raise ValueError("Exact-duplicate group IDs cross the split")

    val = frame.loc[frame["split"] == "validation"].reset_index(drop=True)
    if len(val) != 3984:
        raise ValueError(f"Expected 3,984 validation images; found {len(val)}")
    if val["class"].isna().any() or not set(val["class"]).issubset(CLASSES):
        raise ValueError("Unexpected validation class")

    pipeline = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    dataset = FacialFeaturesDataset(val, image_dir, pipeline)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("classes") != CLASSES:
        raise ValueError("Checkpoint class order differs from train_clean.py")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    actual, predicted = [], []
    print(f"Device: {device}; validation images: {len(dataset):,}", flush=True)
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            total_loss += criterion(logits, targets.to(device)).item()
            actual.extend(targets.tolist())
            predicted.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = {
        "checkpoint": "best_model_clean.pth",
        "split": "clean_train_validation_split.csv",
        "validation_count": len(dataset),
        "validation_cross_entropy_loss": total_loss / len(dataset),
        "validation_accuracy": accuracy_score(actual, predicted),
        "validation_macro_f1": f1_score(actual, predicted, average="macro", zero_division=0),
        "note": "Loss is calculated for the selected Epoch 4 checkpoint only; historical validation losses were not saved.",
    }
    if abs(metrics["validation_accuracy"] - 0.9827) > 0.0001:
        raise RuntimeError("Selected checkpoint accuracy did not reproduce 0.9827")
    if abs(metrics["validation_macro_f1"] - 0.9822) > 0.0001:
        raise RuntimeError("Selected checkpoint macro F1 did not reproduce 0.9822")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print("Validation cross-entropy loss:", f'{metrics["validation_cross_entropy_loss"]:.6f}')
    print("Validation accuracy:", f'{metrics["validation_accuracy"]:.6f}')
    print("Validation macro F1:", f'{metrics["validation_macro_f1"]:.6f}')
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
