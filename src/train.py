from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from .config import Config, config_to_dict, load_config
from .data import DatasetBundle, prepare_dataset_bundle
from .input_pipeline import build_tf_dataset
from .utils import create_run_dir, save_json, save_text, set_global_seed, setup_logging


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


def plot_training_history(
    history: tf.keras.callbacks.History,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def train(config: Config, run_dir: Path) -> Tuple[Model, DatasetBundle, dict]:
    logs_dir = run_dir / "logs"
    plots_dir = run_dir / "plots"
    metrics_dir = run_dir / "metrics"
    models_dir = run_dir / "models"
    artifacts_dir = run_dir / "artifacts"

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

    history = model.fit(
        train_tf.dataset,
        validation_data=val_tf.dataset,
        epochs=config.training.epochs,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
        steps_per_epoch=train_tf.steps_per_epoch,
        validation_steps=val_tf.steps_per_epoch,
        verbose=1,
    )

    plot_training_history(history, plots_dir / "training_history.png")
    logger.info("Training history plot saved to %s", plots_dir / "training_history.png")

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
    predicted_labels = np.argmax(predictions, axis=1)
    true_labels = dataset_bundle.test.labels

    average_strategy = "binary" if len(dataset_bundle.class_names) == 2 else "macro"
    precision = precision_score(true_labels, predicted_labels, average=average_strategy, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average=average_strategy, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average=average_strategy, zero_division=0)

    report = classification_report(
        true_labels,
        predicted_labels,
        target_names=dataset_bundle.class_names,
    )
    save_text(report, metrics_dir / "classification_report.txt")

    cm = confusion_matrix(true_labels, predicted_labels)
    plot_confusion_matrix(cm, dataset_bundle.class_names, plots_dir / "confusion_matrix.png")

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "class_names": dataset_bundle.class_names,
    }
    save_json(metrics, metrics_dir / "metrics.json")

    metadata = {
        "config": config_to_dict(config),
        "class_to_index": dataset_bundle.class_to_index,
        "run_dir": str(run_dir),
    }
    save_json(metadata, run_dir / "metadata.json")

    final_model_path = models_dir / "final_model.keras"
    model.save(final_model_path)
    LOGGER.info("Final model saved to %s", final_model_path)

    registry_dir = config.artifacts.model_registry
    registry_dir.mkdir(parents=True, exist_ok=True)
    export_path = registry_dir / "best_model.keras"
    tf.keras.models.save_model(model, export_path, overwrite=True)
    LOGGER.info("Model exported to %s", export_path)

    return model, dataset_bundle, metrics


def main(cli_args: list[str] | None = None) -> None:
    args = parse_args(cli_args)
    config = load_config(args.config.resolve())
    run_dir = create_run_dir(config.artifacts.root, run_name=args.run_name)
    train(config, run_dir)


if __name__ == "__main__":
    main()
