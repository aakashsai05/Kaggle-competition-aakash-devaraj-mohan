import pandas as pd
from pathlib import Path
from PIL import Image
from collections import Counter

image_dir = Path("train/train")
image_paths = sorted(image_dir.glob("*.jpg"))

widths = []
heights = []
size_counter = Counter()
corrupted = []

for path in image_paths:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            w, h = img.size
        widths.append(w)
        heights.append(h)
        size_counter[(w, h)] += 1
    except Exception:
        corrupted.append(path.name)

output_dir = Path("01_Dataset_Exploration/results")
output_dir.mkdir(parents=True, exist_ok=True)

summary = {
    "total_images": len(image_paths),
    "corrupted_images": len(corrupted),
    "min_width": min(widths) if widths else None,
    "max_width": max(widths) if widths else None,
    "min_height": min(heights) if heights else None,
    "max_height": max(heights) if heights else None,
    "unique_image_sizes": len(size_counter),
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(output_dir / "image_quality_summary.csv", index=False)

size_df = pd.DataFrame(
    [
        {"width": w, "height": h, "count": c}
        for (w, h), c in size_counter.most_common()
    ]
)
size_df.to_csv(output_dir / "image_size_counts.csv", index=False)

with open(output_dir / "corrupted_images.txt", "w", encoding="utf-8") as f:
    if corrupted:
        for name in corrupted:
            f.write(name + "\n")
    else:
        f.write("No corrupted images found.\n")

print("Total images checked:", len(image_paths))
print("Corrupted images:", len(corrupted))
print("Minimum width:", min(widths) if widths else None)
print("Maximum width:", max(widths) if widths else None)
print("Minimum height:", min(heights) if heights else None)
print("Maximum height:", max(heights) if heights else None)
print("Unique image sizes:", len(size_counter))
print("Most common sizes:")
for (w, h), c in size_counter.most_common(10):
    print(f"{w}x{h}: {c}")

print("Quality check files saved successfully.")

