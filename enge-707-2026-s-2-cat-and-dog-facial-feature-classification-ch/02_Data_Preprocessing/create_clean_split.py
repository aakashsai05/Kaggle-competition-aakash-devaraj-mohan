
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "02_Data_Preprocessing" / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)


labels = pd.read_csv(ROOT / "train_labels.csv")
image_dir = ROOT / "train" / "train"

df = labels[
    labels["image_id"].apply(
        lambda x: (image_dir / f"{int(x):05d}.jpg").is_file()
    )
].copy()


duplicates = pd.read_csv(
    ROOT / "01_Dataset_Exploration"
    / "results" / "duplicate_images.csv"
)

duplicate_map = dict(
    zip(duplicates["image_id"], duplicates["duplicate_group"])
)


def get_group(image_id):
    if image_id in duplicate_map:
        return f"duplicate_{duplicate_map[image_id]}"
    return f"image_{image_id}"

df["group_id"] = df["image_id"].apply(get_group)


if (df.groupby("group_id")["class"].nunique() > 1).any():
    raise ValueError("A duplicate group has conflicting labels")


groups = df.groupby("group_id", as_index=False).agg(
    class_name=("class", "first")
)

train_groups, val_groups = train_test_split(
    groups,
    test_size=0.20,
    random_state=42,
    stratify=groups["class_name"]
)

train_ids = set(train_groups["group_id"])
val_ids = set(val_groups["group_id"])

df["split"] = df["group_id"].apply(
    lambda g: "train" if g in train_ids else "validation"
)


train_set = set(df.loc[df["split"] == "train", "group_id"])
val_set = set(df.loc[df["split"] == "validation", "group_id"])

overlap = train_set.intersection(val_set)

assert len(overlap) == 0, "Duplicate leakage detected!"


df[["image_id", "class", "split", "group_id"]].to_csv(
    OUTPUT / "clean_train_validation_split.csv",
    index=False
)

print("Total images:", len(df))
print("Training images:", (df["split"] == "train").sum())
print("Validation images:", (df["split"] == "validation").sum())
print("Duplicate groups crossing splits:", len(overlap))
print()
print("Class distribution:")
print(pd.crosstab(df["class"], df["split"]))
print()
print("Duplicate-safe split saved successfully.")
