"""Compare horizontal-flip TTA against multi-scale TTA."""

import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "03_Model_Development"))
from train_clean import CLASSES


class ValidationDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)
        self.folder = ROOT / "train" / "train"
        self.mapping = {name: i for i, name in enumerate(CLASSES)}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = self.folder / f"{int(row['image_id']):05d}.jpg"

        with Image.open(path) as img:
            image = img.convert("RGB")

            def prepare(size):
                transform = transforms.Compose([
                    transforms.Resize((size, size)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225),
                    ),
                ])
                return transform(image)

            return (
                prepare(224),
                prepare(256),
                self.mapping[row["class"]],
            )


def main():
    split = pd.read_csv(
        ROOT / "02_Data_Preprocessing" / "results"
        / "clean_train_validation_split.csv"
    )

    validation = split[
        split["split"] == "validation"
    ].reset_index(drop=True)

    assert len(validation) == 3984

    loader = DataLoader(
        ValidationDataset(validation),
        batch_size=32,
        shuffle=False,
        num_workers=0,
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

    assert checkpoint["classes"] == CLASSES

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()

    actual = []
    predictions = {
        "Horizontal Flip TTA": [],
        "Multi-scale TTA 50-50": [],
        "Multi-scale TTA 75-25": [],
    }

    print(f"Device: {device}", flush=True)
    print("Evaluating multi-scale TTA...", flush=True)

    with torch.inference_mode():
        for batch, (x224, x256, labels) in enumerate(loader, 1):
            x224 = x224.to(device)
            x256 = x256.to(device)

            p224 = (
                torch.softmax(model(x224), dim=1)
                + torch.softmax(model(torch.flip(x224, [3])), dim=1)
            ) / 2

            p256 = (
                torch.softmax(model(x256), dim=1)
                + torch.softmax(model(torch.flip(x256, [3])), dim=1)
            ) / 2

            results = {
                "Horizontal Flip TTA": p224,
                "Multi-scale TTA 50-50": 0.5 * p224 + 0.5 * p256,
                "Multi-scale TTA 75-25": 0.75 * p224 + 0.25 * p256,
            }

            actual.extend(labels.tolist())

            for name, probabilities in results.items():
                predictions[name].extend(
                    probabilities.argmax(1).cpu().tolist()
                )

            if batch % 40 == 0:
                print(
                    f"Processed {len(actual):,}/3,984",
                    flush=True,
                )

    rows = []
    for name, predicted in predictions.items():
        rows.append({
            "method": name,
            "accuracy": accuracy_score(actual, predicted),
            "macro_f1": f1_score(
                actual, predicted, average="macro", zero_division=0
            ),
        })

    comparison = pd.DataFrame(rows)

    if abs(comparison.iloc[0]["macro_f1"] - 0.9855) > 0.0001:
        raise RuntimeError("Existing TTA result did not reproduce")

    output = (
        ROOT / "05_Model_Evaluation" / "results"
        / "multiscale_tta_comparison.csv"
    )
    comparison.to_csv(output, index=False)

    print("\nMULTI-SCALE TTA RESULTS")
    print(comparison.to_string(index=False))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
