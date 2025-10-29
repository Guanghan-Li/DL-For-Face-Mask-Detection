from __future__ import annotations

from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def load_image(
    path: tf.Tensor,
    label: tf.Tensor,
    image_size: Tuple[int, int],
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Read an image from disk and resize it to the desired shape."""
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.BILINEAR)
    image = tf.cast(image, tf.float32)
    return image, label


def build_data_augmentation(seed: int | None = None) -> tf.keras.Sequential:
    """Create data augmentation pipeline mirroring ImageDataGenerator settings."""
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=seed),
            layers.RandomRotation(0.1, fill_mode="nearest", seed=seed),
            layers.RandomTranslation(0.1, 0.1, fill_mode="nearest", seed=seed),
            layers.RandomZoom(0.2, fill_mode="nearest", seed=seed),
            layers.RandomBrightness(factor=0.2, value_range=(0.0, 255.0), seed=seed),
        ],
        name="augmentation",
    )


def apply_augmentation(
    image: tf.Tensor,
    label: tf.Tensor,
    augmenter: tf.keras.Sequential,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply data augmentation to an image/label pair."""
    image = augmenter(image, training=True)
    return image, label


def apply_preprocessing(
    image: tf.Tensor,
    label: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply MobileNetV2 preprocessing to the image tensor."""
    image = preprocess_input(image)
    return image, label
