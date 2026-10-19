
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "02_Data_Preprocessing" / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(ROOT / "train_labels.csv")
image_dir = ROOT / "train" / "train"

df = df[
    df["image_id"].apply(
        lambda image_id: (
            image_dir / f"{int(image_id):05d}.jpg"
        ).is_file()
    )
].copy()


train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["class"]
)

train_df = train_df.copy()
val_df = val_df.copy()

train_df["split"] = "train"
val_df["split"] = "validation"

splits = pd.concat([train_df, val_df])
splits = splits.sort_values("image_id")
splits.to_csv(OUTPUT / "train_validation_split.csv", index=False)

class_counts = pd.crosstab(
    splits["class"],
    splits["split"]
)

class_counts.to_csv(OUTPUT / "split_class_distribution.csv")


train_ids = set(train_df["image_id"])
val_ids = set(val_df["image_id"])

overlap = train_ids.intersection(val_ids)


duplicate_path = (
    ROOT / "01_Dataset_Exploration"
    / "results" / "duplicate_images.csv"
)

duplicates = pd.read_csv(duplicate_path)

split_map = dict(zip(splits["image_id"], splits["split"]))

duplicates["split"] = duplicates["image_id"].map(split_map)

cross_split_groups = 0

for group_id, group in duplicates.groupby("duplicate_group"):
    if group["split"].dropna().nunique() > 1:
        cross_split_groups += 1

print("Total available images:", len(df))
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Overlapping image IDs:", len(overlap))
print("Exact duplicate groups crossing splits:", cross_split_groups)
print()
print("Class distribution:")
print(class_counts)
print()
print("Preprocessing audit completed.")


summary = pd.DataFrame([{
    "available_images": len(df),
    "training_images": len(train_df),
    "validation_images": len(val_df),
    "overlapping_image_ids": len(overlap),
    "duplicate_groups_crossing_splits": cross_split_groups
}])

summary.to_csv(OUTPUT / "preprocessing_summary.csv", index=False)

