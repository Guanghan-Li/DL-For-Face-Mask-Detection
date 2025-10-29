from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from .config import load_config
from .utils import save_json, setup_logging


LOGGER = logging.getLogger("mask_detector")


def parse_args(cli_args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with the trained mask detector.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to a saved Keras model.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the image to classify.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional path to metadata.json generated during training.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Config for image size fallback if metadata is absent.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the prediction JSON payload.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("artifacts/predict_logs"),
        help="Directory for prediction logs.",
    )
    return parser.parse_args(cli_args)


def load_class_names(metadata_path: Path | None) -> List[str]:
    if metadata_path and metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload: Dict[str, object] = json.load(handle)
        class_to_index = payload.get("class_to_index", {})
        if class_to_index:
            sorted_items = sorted(class_to_index.items(), key=lambda item: item[1])
            return [name for name, _ in sorted_items]
    return ["without_mask", "with_mask"]


def preprocess_image(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    image_bytes = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    image = tf.expand_dims(image, axis=0)
    return image.numpy()


def main(cli_args: list[str] | None = None) -> None:
    args = parse_args(cli_args)
    logger = setup_logging(args.log_dir)

    config = load_config(args.config.resolve())
    image_size = tuple(config.data.image_size)

    class_names = load_class_names(args.metadata)
    logger.info("Loaded class names: %s", class_names)

    if not args.image.exists():
        raise FileNotFoundError(f"Image file not found: {args.image}")

    model = tf.keras.models.load_model(args.model_path)
    logger.info("Loaded model from %s", args.model_path)

    inputs = preprocess_image(args.image, image_size)
    predictions = model.predict(inputs, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_label = class_names[predicted_index]
    confidence = float(predictions[predicted_index])

    logger.info("Prediction for %s: %s (%.2f%%)", args.image, predicted_label, confidence * 100)

    result = {
        "image": str(args.image),
        "predicted_label": predicted_label,
        "confidence": confidence,
        "class_names": class_names,
        "raw_probabilities": predictions.tolist(),
    }

    if args.output:
        save_json(result, args.output)
        logger.info("Prediction saved to %s", args.output)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
