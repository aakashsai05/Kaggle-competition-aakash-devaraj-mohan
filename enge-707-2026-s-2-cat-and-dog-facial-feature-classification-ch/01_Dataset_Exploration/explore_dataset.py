import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load training labels
df = pd.read_csv("train_labels.csv")

# Find available training images
image_dir = Path("train/train")
image_ids = {int(p.stem) for p in image_dir.glob("*.jpg")}

# Match images to labels
df = df[df["image_id"].isin(image_ids)]

# Count images in each class
counts = df["class"].value_counts()

# Create results directory
output_dir = Path("01_Dataset_Exploration/results")
output_dir.mkdir(parents=True, exist_ok=True)

# Save class distribution CSV
counts.rename_axis("class").reset_index(
    name="image_count"
).to_csv(output_dir / "class_counts.csv", index=False)

# Create class distribution graph
plt.figure(figsize=(11, 6))
bars = plt.bar(counts.index, counts.values)

plt.title("Class Distribution of Training Images")
plt.xlabel("Facial Feature Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=30)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 30,
        str(int(bar.get_height())),
        ha="center"
    )

plt.ylim(0, counts.max() * 1.12)
plt.tight_layout()
plt.savefig(output_dir / "class_distribution.png", dpi=300)
plt.close()

print("Total matching images:", len(df))
print(counts.to_string())
print("Graph and class counts saved successfully.")

