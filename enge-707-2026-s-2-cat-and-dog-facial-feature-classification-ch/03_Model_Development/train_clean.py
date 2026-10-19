"""ENGE707 Kaggle: baseline 8-class cat/dog facial-feature classifier.

Place this file beside train_labels.csv and the extracted train/ directory.
The image archive has entries like train/train/00001.jpg, so after extraction
this script expects ./train/train/00001.jpg.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CLASSES = ["CatEye", "CatFace", "CatMouth", "CatNose", "DogEye", "DogFace", "DogMouth", "DogNose"]
ROOT = Path(__file__).resolve().parent.parent


class FacialFeaturesDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_dir: Path, transform):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.class_to_idx = {name: i for i, name in enumerate(CLASSES)}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = self.image_dir / f"{int(row['image_id']):05d}.jpg"
        with Image.open(path) as img:
            image = img.convert("RGB")
            image = self.transform(image)
        return image, self.class_to_idx[row["class"]]


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0003)
    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    labels_csv = ROOT / "train_labels.csv"
    image_dir = ROOT / "train" / "train"
    if not labels_csv.is_file():
        raise FileNotFoundError(f"Missing labels CSV: {labels_csv}")
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"Image folder not found: {image_dir}\n"
            "Extract train.zip into this script's folder first."
        )

    df = pd.read_csv(labels_csv)
    if not {"image_id", "class"}.issubset(df.columns):
        raise ValueError("train_labels.csv must contain 'image_id' and 'class' columns")
    if df["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id values in train_labels.csv")
    unknown = set(df["class"].dropna().unique()) - set(CLASSES)
    if unknown or df["class"].isna().any():
        raise ValueError(f"Unexpected or missing labels: {unknown}")

    paths = [image_dir / f"{int(i):05d}.jpg" for i in df["image_id"]]
    available = pd.Series([p.is_file() for p in paths], index=df.index)
    present = df.loc[available].copy()
    print(f"Labels in CSV: {len(df):,}")
    print(f"Matching image files: {len(present):,}")
    print(f"Labels without images: {(~available).sum():,}")
    if present.empty or present["class"].nunique() != len(CLASSES):
        raise RuntimeError("Not enough matching images across all eight classes")

    # Load the duplicate-safe split from Task 2
    split_path = ROOT / "02_Data_Preprocessing" / "results" / "clean_train_validation_split.csv"
    split_df = pd.read_csv(split_path)

    if split_df["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs in clean split")
    if set(split_df["image_id"]) != set(present["image_id"]):
        raise ValueError("Clean split does not match available images")
    if set(split_df["split"]) != {"train", "validation"}:
        raise ValueError("Invalid split assignments")

    expected_labels = present.set_index("image_id")["class"].sort_index()
    saved_labels = split_df.set_index("image_id")["class"].sort_index()
    if not expected_labels.equals(saved_labels):
        raise ValueError("Clean split labels do not match training labels")

    train_rows = split_df[split_df["split"] == "train"]
    val_rows = split_df[split_df["split"] == "validation"]

    if set(train_rows["group_id"]) & set(val_rows["group_id"]):
        raise ValueError("Duplicate groups cross the split")

    train_df = present[present["image_id"].isin(train_rows["image_id"])].copy()
    val_df = present[present["image_id"].isin(val_rows["image_id"])].copy()
    print(f"Training images: {len(train_df):,} | Validation images: {len(val_df):,}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    train_data = FacialFeaturesDataset(train_df, image_dir, train_transform)
    val_data = FacialFeaturesDataset(val_df, image_dir, val_transform)
    # num_workers=0 avoids Windows multiprocessing issues in a first-time setup.
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cpu":
        print("CPU training may be slow; an NVIDIA CUDA GPU or cloud GPU will be faster.")

    # Pretrained weights are downloaded automatically on the first run.
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.layer4.parameters():
        parameter.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()
    best_f1 = -1.0
    output_path = ROOT / "03_Model_Development" / "checkpoints" / "best_model_clean.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        train_examples = 0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(targets)
            train_examples += len(targets)

        model.eval()
        actual, predicted = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                logits = model(images.to(device))
                predicted.extend(logits.argmax(dim=1).cpu().tolist())
                actual.extend(targets.tolist())

        accuracy = accuracy_score(actual, predicted)
        macro_f1 = f1_score(actual, predicted, average="macro", zero_division=0)
        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={running_loss / train_examples:.4f}, "
            f"val_accuracy={accuracy:.4f}, val_macro_F1={macro_f1:.4f}",
            flush=True,
        )
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            torch.save({"state_dict": model.state_dict(), "classes": CLASSES}, output_path)
            print(f"  Saved best checkpoint: {output_path.name}")

    (ROOT / "03_Model_Development" / "class_mapping.json").write_text(
        json.dumps({name: idx for idx, name in enumerate(CLASSES)}, indent=2),
        encoding="utf-8",
    )
    print(f"Done! Best validation macro F1: {best_f1:.4f}")


if __name__ == "__main__":
    run()


