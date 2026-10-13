import math

from matplotlib import pyplot as plt


def plot_metrics(history: dict[str, list[float]], save_path: str | None = None, show: bool = True):
    """Grafica loss y perplexity de train/val a partir de un dict con 'loss' y 'val_loss'."""
    if "loss" not in history or "val_loss" not in history:
        raise ValueError("history necesita las claves 'loss' y 'val_loss'")

    epochs = list(range(1, len(history["loss"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["loss"], label="Train Loss", color="#0284c7", linewidth=2)
    ax1.plot(epochs, history["val_loss"], label="Validation Loss", color="#f43f5e", linewidth=2, linestyle="--")
    ax1.set_title("Cross-Entropy Loss per Epoch", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # ppl = exp(loss), el min evita overflow con loss muy altos
    train_ppl = [math.exp(min(l, 20.0)) for l in history["loss"]]
    val_ppl = [math.exp(min(l, 20.0)) for l in history["val_loss"]]

    ax2.plot(epochs, train_ppl, label="Train Perplexity", color="#0284c7", linewidth=2)
    ax2.plot(epochs, val_ppl, label="Validation Perplexity", color="#f43f5e", linewidth=2, linestyle="--")
    ax2.set_title("Perplexity per Epoch", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Perplexity", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()


if __name__ == "__main__":
    # datos de ejemplo, no es un entrenamiento real
    sample_history = {
        "loss": [3.45, 2.91, 2.45, 2.12, 1.89, 1.72],
        "val_loss": [3.55, 3.05, 2.62, 2.35, 2.18, 2.09],
    }
    plot_metrics(sample_history, save_path="metrics_plot.png", show=False)
    print("guardado metrics_plot.png")