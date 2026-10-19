import hashlib
from pathlib import Path
from collections import defaultdict
from PIL import Image
import pandas as pd

image_dir = Path("train/train")
output_dir = Path("01_Dataset_Exploration/results")
output_dir.mkdir(parents=True, exist_ok=True)

labels = pd.read_csv("train_labels.csv")
label_map = dict(zip(labels["image_id"], labels["class"]))

hash_groups = defaultdict(list)
small_images = []

for path in sorted(image_dir.glob("*.jpg")):
    image_id = int(path.stem)
    label = label_map.get(image_id, "Unknown")

    with Image.open(path) as img:
        width, height = img.size

    if width < 32 or height < 32:
        small_images.append({
            "image_id": image_id,
            "class": label,
            "width": width,
            "height": height
        })

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)

    hash_groups[hasher.hexdigest()].append((image_id, label))

duplicate_groups = [
    group for group in hash_groups.values()
    if len(group) > 1
]

duplicate_rows = []
for group_num, group in enumerate(duplicate_groups, start=1):
    for image_id, label in group:
        duplicate_rows.append({
            "duplicate_group": group_num,
            "image_id": image_id,
            "class": label
        })

pd.DataFrame(
    small_images,
    columns=["image_id", "class", "width", "height"]
).to_csv(output_dir / "small_images.csv", index=False)

pd.DataFrame(
    duplicate_rows,
    columns=["duplicate_group", "image_id", "class"]
).to_csv(output_dir / "duplicate_images.csv", index=False)

conflicting_groups = sum(
    len({label for _, label in group}) > 1
    for group in duplicate_groups
)

print("Images checked:", sum(len(g) for g in hash_groups.values()))
print("Images smaller than 32 pixels in either dimension:", len(small_images))
print("Exact duplicate groups:", len(duplicate_groups))
print("Extra duplicate copies:", sum(len(g) - 1 for g in duplicate_groups))
print("Duplicate groups with conflicting labels:", conflicting_groups)
print("Duplicate and small-image reports saved.")

