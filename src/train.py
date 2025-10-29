from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import auc, classification_report, confusion_matrix, precision_recall_curve, precision_score, recall_score, roc_curve
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from .callbacks import LearningRateLogger
from .config import Config, config_to_dict, load_config
from .data import DatasetBundle, DatasetSplit, prepare_dataset_bundle
from .input_pipeline import build_tf_dataset
from .utils import create_run_dir, save_json, save_text, set_global_seed, setup_logging
from .visualization import (
    plot_class_distribution,
    plot_confusion_matrix_enhanced,
    plot_learning_rate_schedule,
    plot_per_class_metrics,
    plot_pr_curve,
    plot_roc_curve,
    plot_sample_predictions,
    plot_training_history_enhanced,
    render_interactive_dashboard,
    set_plot_style,
)


LOGGER = logging.getLogger("mask_detector")


def parse_args(cli_args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the face-mask detection model.")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"), help="Path to YAML config.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional suffix for the artifact run folder.")
    return parser.parse_args(cli_args)


def build_model(config: Config, image_size: Tuple[int, int], num_classes: int) -> Model:
    inputs = Input(shape=(*image_size, 3), name="input_image")
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*image_size, 3))
    base_model.trainable = False

    features = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(features)
    x = Dense(config.training.dense_units, activation="relu")(x)
    x = Dropout(config.training.dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs, name="mask_detector")
    optimizer = Adam(learning_rate=config.training.learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history_csv(history_df: pd.DataFrame, output_path: Path, logger: logging.Logger) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(output_path, index=False)
    logger.info("Epoch metrics saved to %s", output_path)


def _compute_class_counts(labels: np.ndarray, class_names: List[str]) -> Dict[str, int]:
    counts = {name: 0 for name in class_names}
    unique, freqs = np.unique(labels, return_counts=True)
    for idx, count in zip(unique, freqs):
        counts[class_names[int(idx)]] = int(count)
    return counts


def _collect_sample_predictions(
    split: DatasetSplit,
    class_names: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    sample_count: int,
    seed: int,
) -> List[Dict[str, object]]:
    total = len(split.files)
    if total == 0:
        return []
    count = min(sample_count, total)
    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=count, replace=False)
    samples: List[Dict[str, object]] = []
    for idx in indices:
        file_path = split.files[idx]
        pred_idx = int(y_pred[idx])
        confidence = float(y_proba[idx]) if y_proba.ndim == 1 else float(y_proba[idx, pred_idx])
        samples.append(
            {
                "image_path": str(file_path),
                "true_label": class_names[int(y_true[idx])],
                "pred_label": class_names[pred_idx],
                "confidence": confidence,
                "is_correct": bool(y_true[idx] == y_pred[idx]),
            }
        )
    return samples


def assemble_callbacks(
    models_dir: Path,
    run_dir: Path,
    config: Config,
) -> Tuple[List[tf.keras.callbacks.Callback], LearningRateLogger]:
    checkpoint_cb = ModelCheckpoint(
        filepath=models_dir / "best_model.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1,
    )
    early_stopping_cb = EarlyStopping(
        monitor="val_loss",
        patience=config.training.early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr_cb = ReduceLROnPlateau(
        monitor="val_loss",
        factor=config.training.reduce_lr_factor,
        patience=config.training.reduce_lr_patience,
        min_lr=config.training.min_lr,
        verbose=1,
    )
    lr_logger = LearningRateLogger()
    tensorboard_dir = run_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_cb = TensorBoard(
        log_dir=tensorboard_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
    )
    callbacks: List[tf.keras.callbacks.Callback] = [
        checkpoint_cb,
        early_stopping_cb,
        reduce_lr_cb,
        lr_logger,
        tensorboard_cb,
    ]
    return callbacks, lr_logger


def train(config: Config, run_dir: Path) -> Tuple[Model, DatasetBundle, dict]:
    logs_dir = run_dir / "logs"
    plots_dir = run_dir / "plots"
    metrics_dir = run_dir / "metrics"
    models_dir = run_dir / "models"
    artifacts_dir = run_dir / "artifacts"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(logs_dir)
    logger.info("Starting training run in %s", run_dir)

    set_global_seed(config.seed)
    logger.info("Global seed set to %d", config.seed)

    dataset_bundle = prepare_dataset_bundle(
        data_root=config.data.root,
        artifacts_dir=artifacts_dir,
        validation_split=config.data.validation_split,
        test_split=config.data.test_split,
        seed=config.seed,
        logger=logger,
    )

    train_tf = build_tf_dataset(
        split=dataset_bundle.train,
        image_size=config.data.image_size,
        batch_size=config.data.batch_size,
        shuffle=True,
        augment=True,
        seed=config.seed,
    )
    val_tf = build_tf_dataset(
        split=dataset_bundle.validation,
        image_size=config.data.image_size,
        batch_size=config.data.batch_size,
        shuffle=False,
        augment=False,
        seed=config.seed,
    )
    test_tf = build_tf_dataset(
        split=dataset_bundle.test,
        image_size=config.data.image_size,
        batch_size=config.data.batch_size,
        shuffle=False,
        augment=False,
        seed=config.seed,
    )

    logger.info(
        "TF datasets prepared | train steps=%d, val steps=%d, test steps=%d",
        train_tf.steps_per_epoch,
        val_tf.steps_per_epoch,
        test_tf.steps_per_epoch,
    )

    model = build_model(config, config.data.image_size, num_classes=len(dataset_bundle.class_names))
    logger.info("Model compiled with %d parameters", model.count_params())

    models_dir.mkdir(parents=True, exist_ok=True)

    callbacks, lr_logger = assemble_callbacks(models_dir, run_dir, config)
    logger.info("TensorBoard logs directory: %s", run_dir / "tensorboard")

    history = model.fit(
        train_tf.dataset,
        validation_data=val_tf.dataset,
        epochs=config.training.epochs,
        callbacks=callbacks,
        steps_per_epoch=train_tf.steps_per_epoch,
        validation_steps=val_tf.steps_per_epoch,
        verbose=1,
    )

    history_df = pd.DataFrame(history.history)
    history_df["epoch"] = np.arange(1, len(history_df) + 1)
    history_csv = metrics_dir / "epoch_metrics.csv"
    save_history_csv(history_df, history_csv, logger)
    best_epoch = int(history_df["val_loss"].idxmin()) if "val_loss" in history_df else len(history_df) - 1

    logger.info("Evaluating on test dataset")
    test_loss, test_accuracy = model.evaluate(
        test_tf.dataset,
        steps=test_tf.steps_per_epoch,
        verbose=1,
    )
    logger.info("Test loss=%.4f, accuracy=%.4f", test_loss, test_accuracy)

    predictions = model.predict(
        test_tf.dataset,
        steps=test_tf.steps_per_epoch,
        verbose=1,
    )
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = dataset_bundle.test.labels

    average_strategy = "binary" if len(dataset_bundle.class_names) == 2 else "macro"
    precision = precision_score(true_labels, predicted_labels, average=average_strategy, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average=average_strategy, zero_division=0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=dataset_bundle.class_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        true_labels,
        predicted_labels,
        target_names=dataset_bundle.class_names,
        zero_division=0,
    )
    report_path = metrics_dir / "classification_report.txt"
    save_text(report_text, report_path)
    logger.info("Classification report saved to %s", report_path)

    cm = confusion_matrix(true_labels, predicted_labels, labels=range(len(dataset_bundle.class_names)))

    per_cls_df = (
        pd.DataFrame({k: v for k, v in report.items() if k in dataset_bundle.class_names})
        .T.reset_index()
        .rename(columns={"index": "class"})
    )
    per_cls_csv = metrics_dir / "per_class_metrics.csv"
    per_cls_df.to_csv(per_cls_csv, index=False)
    logger.info("Per-class metrics saved to %s", per_cls_csv)

    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True).clip(min=1.0)
    cm_norm = cm_norm / row_sums
    cm_csv = metrics_dir / "confusion_matrix_normalized.csv"
    pd.DataFrame(cm_norm, index=dataset_bundle.class_names, columns=dataset_bundle.class_names).to_csv(cm_csv)
    logger.info("Normalized confusion matrix saved to %s", cm_csv)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "class_names": dataset_bundle.class_names,
    }

    roc_auc = None
    pr_auc = None
    if len(dataset_bundle.class_names) == 2:
        pos_scores = predictions[:, 1] if predictions.shape[1] > 1 else predictions[:, 0]
        fpr, tpr, _ = roc_curve(true_labels, pos_scores)
        precision_curve, recall_curve, _ = precision_recall_curve(true_labels, pos_scores)
        roc_auc = float(auc(fpr, tpr))
        pr_auc = float(auc(recall_curve, precision_curve))
        metrics["roc_auc"] = roc_auc
        metrics["pr_auc"] = pr_auc

    metrics_path = metrics_dir / "metrics.json"
    save_json(metrics, metrics_path)
    logger.info("Metrics JSON saved to %s", metrics_path)

    metadata = {
        "config": config_to_dict(config),
        "class_to_index": dataset_bundle.class_to_index,
        "run_dir": str(run_dir),
        "artifacts": {
            "history_csv": str(history_csv),
            "per_class_csv": str(per_cls_csv),
            "confusion_matrix_csv": str(cm_csv),
        },
    }
    metadata_path = run_dir / "metadata.json"
    save_json(metadata, metadata_path)
    logger.info("Metadata saved to %s", metadata_path)

    if config.visualization.enabled:
        set_plot_style(config.visualization)
        history_plot_path = plots_dir / "training_history_enhanced.png"
        plot_training_history_enhanced(history_df, best_epoch, lr_logger.learning_rates, history_plot_path, config.visualization)
        logger.info("Training history plot saved to %s", history_plot_path)

        test_accuracy_rate = float(np.mean(true_labels == predicted_labels))
        cm_plot_path = plots_dir / "confusion_matrix_enhanced.png"
        plot_confusion_matrix_enhanced(
            cm,
            dataset_bundle.class_names,
            test_accuracy_rate,
            cm_plot_path,
            normalize=True,
            config=config.visualization,
        )
        logger.info("Confusion matrix plot saved to %s", cm_plot_path)

        per_class_plot_path = plots_dir / "per_class_metrics.png"
        plot_per_class_metrics(per_cls_df, dataset_bundle.class_names, per_class_plot_path, config.visualization)
        logger.info("Per-class metrics plot saved to %s", per_class_plot_path)

        if config.visualization.include_lr_schedule:
            lr_plot_path = plots_dir / "learning_rate_schedule.png"
            plot_learning_rate_schedule(lr_logger.learning_rates, lr_plot_path, config.visualization)
            logger.info("Learning rate schedule plot saved to %s", lr_plot_path)

        if config.visualization.include_class_distribution:
            split_counts = {
                "train": _compute_class_counts(dataset_bundle.train.labels, dataset_bundle.class_names),
                "val": _compute_class_counts(dataset_bundle.validation.labels, dataset_bundle.class_names),
                "test": _compute_class_counts(dataset_bundle.test.labels, dataset_bundle.class_names),
            }
            class_dist_path = plots_dir / "class_distribution.png"
            plot_class_distribution(split_counts, class_dist_path, config.visualization)
            logger.info("Class distribution plot saved to %s", class_dist_path)

        if len(dataset_bundle.class_names) == 2:
            pos_scores = predictions[:, 1] if predictions.shape[1] > 1 else predictions[:, 0]
            if config.visualization.include_roc_curve:
                roc_path = plots_dir / "roc_curve.png"
                plot_roc_curve(true_labels, pos_scores, roc_path, config.visualization)
                logger.info("ROC curve saved to %s", roc_path)
            if config.visualization.include_pr_curve:
                pr_path = plots_dir / "precision_recall_curve.png"
                plot_pr_curve(true_labels, pos_scores, pr_path, config.visualization)
                logger.info("Precision-recall curve saved to %s", pr_path)

        samples = _collect_sample_predictions(
            dataset_bundle.test,
            dataset_bundle.class_names,
            true_labels,
            predicted_labels,
            predictions,
            config.visualization.sample_predictions_count,
            seed=config.seed,
        )
        if samples:
            sample_path = plots_dir / "sample_predictions.png"
            plot_sample_predictions(samples, sample_path, config.visualization)
            logger.info("Sample predictions grid saved to %s", sample_path)

        dashboard_path = plots_dir / "metrics_dashboard.html"
        if config.visualization.include_dashboard:
            render_interactive_dashboard(
                history_csv=history_csv,
                cm_csv=cm_csv,
                per_cls_csv=per_cls_csv,
                out_html=dashboard_path,
            )
            logger.info("Interactive dashboard saved to %s", dashboard_path)

    final_model_path = models_dir / "final_model.keras"
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(final_model_path)
    logger.info("Final model saved to %s", final_model_path)

    registry_dir = config.artifacts.model_registry
    registry_dir.mkdir(parents=True, exist_ok=True)
    export_path = registry_dir / "best_model.keras"
    tf.keras.models.save_model(model, export_path, overwrite=True)
    logger.info("Model exported to %s", export_path)

    return model, dataset_bundle, metrics


def main(cli_args: list[str] | None = None) -> None:
    args = parse_args(cli_args)
    config = load_config(args.config.resolve())
    run_dir = create_run_dir(config.artifacts.root, run_name=args.run_name)
    train(config, run_dir)


if __name__ == "__main__":
    main()
