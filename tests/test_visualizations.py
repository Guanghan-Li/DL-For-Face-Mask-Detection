from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tensorflow.keras.callbacks import TensorBoard

from src.callbacks import LearningRateLogger
from src.config import ArtifactsConfig, Config, DataConfig, TrainingConfig, VisualizationConfig
from src.train import assemble_callbacks, save_history_csv
from src.visualization import (
    plot_confusion_matrix_enhanced,
    plot_per_class_metrics,
    plot_training_history_enhanced,
    render_interactive_dashboard,
)


def test_epoch_metrics_csv_columns(tmp_path: Path) -> None:
    history_df = pd.DataFrame(
        {
            "loss": [0.9, 0.5],
            "accuracy": [0.6, 0.8],
            "val_loss": [1.0, 0.6],
            "val_accuracy": [0.55, 0.75],
        }
    )
    history_df["epoch"] = [1, 2]
    out_path = tmp_path / "metrics" / "epoch_metrics.csv"
    logger = logging.getLogger("test_history_csv")
    logger.setLevel(logging.INFO)
    save_history_csv(history_df, out_path, logger)

    loaded = pd.read_csv(out_path)
    expected_columns = {"loss", "accuracy", "val_loss", "val_accuracy", "epoch"}
    assert expected_columns.issubset(set(loaded.columns))


def test_confusion_matrix_png_exists(tmp_path: Path) -> None:
    config = VisualizationConfig()
    out_path = tmp_path / "plots" / "confusion_matrix_enhanced.png"
    cm = np.array([[8, 2], [1, 9]])
    plot_confusion_matrix_enhanced(cm, ["with_mask", "without_mask"], test_accuracy=0.85, out_path=out_path, normalize=True, config=config)
    assert out_path.exists()


def test_training_history_plot(tmp_path: Path) -> None:
    config = VisualizationConfig()
    history_df = pd.DataFrame(
        {
            "loss": [0.9, 0.4, 0.3],
            "val_loss": [1.0, 0.5, 0.35],
            "accuracy": [0.6, 0.82, 0.9],
            "val_accuracy": [0.55, 0.78, 0.88],
            "epoch": [1, 2, 3],
        }
    )
    lr_series = [0.001, 0.0008, 0.0005]
    out_path = tmp_path / "plots" / "training_history.png"
    plot_training_history_enhanced(history_df, best_epoch=1, lr_series=lr_series, out_path=out_path, config=config)
    assert out_path.exists()


def test_per_class_metrics_plot(tmp_path: Path) -> None:
    config = VisualizationConfig()
    per_cls_df = pd.DataFrame(
        {
            "class": ["with_mask", "without_mask"],
            "precision": [0.8, 0.85],
            "recall": [0.75, 0.88],
            "f1-score": [0.77, 0.86],
        }
    )
    out_path = tmp_path / "plots" / "per_class_metrics.png"
    plot_per_class_metrics(per_cls_df, ["with_mask", "without_mask"], out_path, config)
    assert out_path.exists()


def test_tensorboard_callback_in_callbacks(tmp_path: Path) -> None:
    data_cfg = DataConfig(root=tmp_path / "data", image_size=(128, 128), validation_split=0.2, test_split=0.2, batch_size=8)
    training_cfg = TrainingConfig(
        epochs=1,
        learning_rate=0.001,
        dense_units=32,
        dropout_rate=0.5,
        early_stopping_patience=1,
        reduce_lr_patience=1,
        reduce_lr_factor=0.5,
        min_lr=1e-6,
    )
    artifacts_cfg = ArtifactsConfig(root=tmp_path / "artifacts", model_registry=tmp_path / "models")
    vis_cfg = VisualizationConfig()
    config = Config(seed=42, data=data_cfg, training=training_cfg, artifacts=artifacts_cfg, visualization=vis_cfg)

    models_dir = tmp_path / "models"
    run_dir = tmp_path / "run"
    callbacks, lr_logger = assemble_callbacks(models_dir, run_dir, config)

    assert any(isinstance(callback, TensorBoard) for callback in callbacks)
    assert isinstance(lr_logger, LearningRateLogger)
    assert (run_dir / "tensorboard").exists()


def test_learning_rate_logger_records_values() -> None:
    tf = pytest.importorskip("tensorflow")
    lr_logger = LearningRateLogger()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1), loss="mse")
    x = np.ones((4, 1), dtype=np.float32)
    y = np.ones((4, 1), dtype=np.float32)
    model.fit(x, y, epochs=3, callbacks=[lr_logger], verbose=0)
    assert len(lr_logger.learning_rates) == 3
    assert all(rate is not None for rate in lr_logger.learning_rates)


def test_dashboard_html_created(tmp_path: Path) -> None:
    history_csv = tmp_path / "metrics" / "epoch_metrics.csv"
    cm_csv = tmp_path / "metrics" / "cm.csv"
    per_cls_csv = tmp_path / "metrics" / "per_class.csv"

    history_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"epoch": [1, 2], "accuracy": [0.6, 0.8], "val_accuracy": [0.55, 0.78], "loss": [0.9, 0.5], "val_loss": [1.0, 0.6]}).to_csv(history_csv, index=False)
    pd.DataFrame([[0.8, 0.2], [0.1, 0.9]], columns=["with_mask", "without_mask"], index=["with_mask", "without_mask"]).to_csv(cm_csv)
    pd.DataFrame({"class": ["with_mask", "without_mask"], "precision": [0.8, 0.85], "recall": [0.75, 0.88], "f1-score": [0.77, 0.86]}).to_csv(per_cls_csv, index=False)

    out_html = tmp_path / "plots" / "metrics_dashboard.html"
    render_interactive_dashboard(history_csv, cm_csv, per_cls_csv, out_html)
    assert out_html.exists()
    assert out_html.stat().st_size > 0
