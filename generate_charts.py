import sys
import logging
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime

from music_gen.main import Config, prepare_data, build_model, train_model
from music_gen.transformer import build_transformer_model, train_model as train_transformer

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("chart_output")
ASSETS_DIR = Path("portfolio/assets")
EPOCHS = 30


def plot_training_curves(
    lstm_history: dict,
    transformer_history: dict,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, ylabel in [
        (axes[0], "loss", "Loss"),
        (axes[1], "accuracy", "Accuracy"),
    ]:
        if metric in lstm_history:
            epochs = range(1, len(lstm_history[metric]) + 1)
            ax.plot(
                epochs,
                lstm_history[metric],
                label="LSTM train",
                color="#1f77b4",
                linewidth=1.5,
            )
            ax.plot(
                epochs,
                lstm_history[f"val_{metric}"],
                label="LSTM val",
                color="#1f77b4",
                linestyle="--",
                linewidth=1.5,
            )
        if metric in transformer_history:
            epochs = range(1, len(transformer_history[metric]) + 1)
            ax.plot(
                epochs,
                transformer_history[metric],
                label="Transformer train",
                color="#ff7f0e",
                linewidth=1.5,
            )
            ax.plot(
                epochs,
                transformer_history[f"val_{metric}"],
                label="Transformer val",
                color="#ff7f0e",
                linestyle="--",
                linewidth=1.5,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "LSTM vs Transformer: Training Curves", fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[+] saved %s", output_path)


def plot_model_comparison(
    lstm_size_mb: float,
    transformer_size_mb: float,
    lstm_params: int,
    transformer_params: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    names = ["LSTM", "Transformer"]
    colors = ["#1f77b4", "#ff7f0e"]

    axes[0].bar(names, [lstm_params, transformer_params], color=colors, width=0.5)
    axes[0].set_ylabel("Parameters")
    axes[0].set_title("Parameter Count")

    axes[1].bar(names, [lstm_size_mb, transformer_size_mb], color=colors, width=0.5)
    axes[1].set_ylabel("Size (MB)")
    axes[1].set_title("Model File Size")

    fig.suptitle("Model Size Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("[+] saved %s", output_path)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    cfg = Config(epochs=EPOCHS)
    logger.info("[*] preparing data...")
    voices, n_notes, x_train, y_train, x_test, y_test = prepare_data(cfg)

    cfg_lstm = Config(epochs=EPOCHS)
    cfg_lstm.model_path = str(OUTPUT_DIR / "charts_lstm.keras")
    cfg_lstm.log_dir = str(OUTPUT_DIR / "chart_logs_lstm")

    logger.info("[*] training LSTM (%d epochs)...", EPOCHS)
    lstm_model = build_model(n_notes, len(voices), cfg_lstm)
    lstm_history = train_model(
        lstm_model, x_train, y_train, x_test, y_test, cfg_lstm
    )

    cfg_trans = Config(epochs=EPOCHS)
    cfg_trans.model_path = str(OUTPUT_DIR / "charts_transformer.keras")
    cfg_trans.log_dir = str(OUTPUT_DIR / "chart_logs_trans")

    logger.info("[*] training Transformer (%d epochs)...", EPOCHS)
    trans_model = build_transformer_model(n_notes, len(voices), cfg_trans)
    trans_history = train_transformer(
        trans_model, x_train, y_train, x_test, y_test, cfg_trans
    )

    lstm_size_mb = Path(cfg_lstm.model_path).stat().st_size / (1024 * 1024)
    trans_size_mb = Path(cfg_trans.model_path).stat().st_size / (1024 * 1024)
    lstm_params = lstm_model.count_params()
    trans_params = trans_model.count_params()

    logger.info("LSTM: %d params, %.2f MB", lstm_params, lstm_size_mb)
    logger.info("Transformer: %d params, %.2f MB", trans_params, trans_size_mb)

    curves_path = OUTPUT_DIR / "training_curves.png"
    plot_training_curves(
        lstm_history.history, trans_history.history, curves_path
    )

    sizes_path = OUTPUT_DIR / "model_comparison.png"
    plot_model_comparison(
        lstm_size_mb, trans_size_mb, lstm_params, trans_params, sizes_path
    )

    for src in [curves_path, sizes_path]:
        dst = ASSETS_DIR / src.name
        dst.write_bytes(src.read_bytes())
        logger.info("[+] copied %s -> %s", src.name, dst)


if __name__ == "__main__":
    main()
