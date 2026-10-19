
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results"
OUTPUT.mkdir(parents=True, exist_ok=True)


df = pd.read_csv(ROOT / "training_history.csv")


plt.figure(figsize=(8, 5))
plt.plot(
    df["epoch"],
    df["train_loss"],
    marker="o",
    linewidth=2
)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("ResNet18 Baseline Training Loss")
plt.xticks(df["epoch"])
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT / "training_loss.png", dpi=300)
plt.close()


plt.figure(figsize=(8, 5))
plt.plot(
    df["epoch"],
    df["val_accuracy"] * 100,
    marker="o",
    label="Validation Accuracy"
)
plt.plot(
    df["epoch"],
    df["val_macro_f1"] * 100,
    marker="s",
    label="Macro F1-score"
)

plt.xlabel("Epoch")
plt.ylabel("Performance (%)")
plt.title("ResNet18 Baseline Validation Performance")
plt.xticks(df["epoch"])
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    OUTPUT / "validation_performance.png",
    dpi=300
)
plt.close()

print("Training graphs generated successfully.")
print("Saved in:", OUTPUT)
