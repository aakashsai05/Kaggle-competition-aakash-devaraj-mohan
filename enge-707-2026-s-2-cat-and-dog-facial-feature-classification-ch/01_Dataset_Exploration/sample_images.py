import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

# Load the training labels
df = pd.read_csv("train_labels.csv")
image_dir = Path("train/train")

# Only include available images
df = df[df["image_id"].apply(
    lambda x: (image_dir / f"{int(x):05d}.jpg").exists()
)]

classes = sorted(df["class"].unique())

# Create a reproducible sample figure
fig, axes = plt.subplots(
    len(classes), 3, figsize=(9, 20)
)

for row, class_name in enumerate(classes):
    samples = df[df["class"] == class_name].sample(
        n=3, random_state=42
    )

    for col, (_, sample) in enumerate(samples.iterrows()):
        image_path = image_dir / f"{int(sample['image_id']):05d}.jpg"

        with Image.open(image_path) as img:
            axes[row, col].imshow(img.convert("RGB"))

        axes[row, col].axis("off")
        axes[row, col].set_title(
            f"{class_name} - {image_path.stem}",
            fontsize=10
        )

plt.suptitle(
    "Representative Training Images Across 8 Classes",
    fontsize=16
)
plt.tight_layout(rect=[0, 0, 1, 0.98])

output_dir = Path("01_Dataset_Exploration/results")
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    output_dir / "sample_images.png",
    dpi=200,
    bbox_inches="tight"
)
plt.close()

print("Sample images saved successfully.")
print("Classes:", ", ".join(classes))
print("Total sample images:", len(classes) * 3)

