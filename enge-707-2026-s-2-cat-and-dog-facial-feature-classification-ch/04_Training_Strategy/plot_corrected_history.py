"""Plot observed training history for the duplicate-safe ResNet18 experiment.

Reads the already-recorded 5-epoch CSV. Does not estimate missing validation loss.
From the competition project directory run:
    python 04_Training_Strategy/plot_corrected_history.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HISTORY = ROOT / "04_Training_Strategy" / "training_history_clean.csv"
OUTPUT = ROOT / "04_Training_Strategy" / "results"


def main():
    if not HISTORY.exists():
        raise FileNotFoundError(HISTORY)
    df = pd.read_csv(HISTORY)
    expected = {"epoch", "train_loss", "val_accuracy", "val_macro_f1"}
    if not expected.issubset(df.columns):
        raise ValueError(f"Missing columns: {expected - set(df.columns)}")
    if list(df["epoch"]) != [1, 2, 3, 4, 5]:
        raise ValueError("Unexpected epoch sequence")
    OUTPUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["epoch"], df["train_loss"], "o-", linewidth=2, label="Training loss")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy training loss", title="Corrected ResNet18 — recorded training loss")
    ax.set_xticks(df["epoch"])
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "corrected_training_loss.png", dpi=190)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["epoch"], 100 * df["val_accuracy"], "o-", linewidth=2, label="Validation accuracy")
    ax.plot(df["epoch"], 100 * df["val_macro_f1"], "s-", linewidth=2, label="Validation macro F1")
    ax.axvline(4, linestyle="--", alpha=0.55, label="Selected checkpoint (epoch 4)")
    ax.set(xlabel="Epoch", ylabel="Percent (%)", title="Corrected ResNet18 — recorded validation performance")
    ax.set_xticks(df["epoch"])
    ax.set_ylim(95, 100)
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "corrected_validation_metrics.png", dpi=190)
    plt.close(fig)
    print("Created:", OUTPUT / "corrected_training_loss.png")
    print("Created:", OUTPUT / "corrected_validation_metrics.png")
    print("Note: No per-epoch validation-loss records exist; no such curve was generated.")


if __name__ == "__main__":
    main()
