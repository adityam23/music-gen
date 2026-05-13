import logging
import json
import time
import numpy as np
import tensorflow as tf

from dataclasses import dataclass
from pathlib import Path

from music_gen.main import Config, prepare_data

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    name: str
    param_count: int
    test_loss: float
    test_accuracy: float
    inference_time: float
    model_size_mb: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "param_count": self.param_count,
            "test_loss": round(self.test_loss, 4),
            "test_accuracy": round(self.test_accuracy, 4),
            "inference_time_s": round(self.inference_time, 4),
            "model_size_mb": round(self.model_size_mb, 2),
        }


def evaluate_model(
    model: tf.keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    name: str,
) -> EvalResult:
    logger.info("--- %s ---", name)

    param_count = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
    logger.info("trainable parameters: %d", param_count)

    t0 = time.perf_counter()
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    inf_time = time.perf_counter() - t0
    logger.info("test loss: %.4f  accuracy: %.4f", loss, accuracy)
    logger.info("inference time: %.2fs", inf_time)

    model_path = Path(f"_eval_{name}.keras")
    model.save(str(model_path))
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    model_path.unlink()
    logger.info("model size: %.2f MB", model_size_mb)

    return EvalResult(
        name=name,
        param_count=param_count,
        test_loss=loss,
        test_accuracy=accuracy,
        inference_time=inf_time,
        model_size_mb=model_size_mb,
    )


def compare(
    lstm_path: str,
    transformer_path: str,
    cfg: Config | None = None,
) -> None:
    if cfg is None:
        cfg = Config()

    _, _, _, _, x_test, y_test = prepare_data(cfg)

    results: list[EvalResult] = []

    lstm_model = tf.keras.models.load_model(lstm_path)
    results.append(evaluate_model(lstm_model, x_test, y_test, "LSTM"))

    transformer_model = tf.keras.models.load_model(transformer_path)
    results.append(evaluate_model(transformer_model, x_test, y_test, "Transformer"))

    logger.info("============ COMPARISON ============")
    for r in results:
        logger.info(
            "%s | params: %6d | accuracy: %.4f | loss: %.4f | size: %.1f MB | time: %.2fs",
            r.name,
            r.param_count,
            r.test_accuracy,
            r.test_loss,
            r.model_size_mb,
            r.inference_time,
        )

    report = {"results": [r.to_dict() for r in results]}
    report_path = Path("comparison_report.json")
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("[+] report saved to %s", report_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare LSTM vs Transformer models.")
    parser.add_argument(
        "--lstm", type=str, default="bach.keras", help="Path to LSTM model"
    )
    parser.add_argument(
        "--transformer",
        type=str,
        default="bach_transformer.keras",
        help="Path to Transformer model",
    )
    parser.add_argument(
        "--data-path", type=str, default="data/F.txt", help="Path to dataset"
    )
    args = parser.parse_args()

    cfg = Config(data_path=args.data_path)
    compare(args.lstm, args.transformer, cfg)
