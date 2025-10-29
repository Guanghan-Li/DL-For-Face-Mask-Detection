from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf

from .data import DatasetSplit
from .preprocessing import apply_augmentation, apply_preprocessing, build_data_augmentation, load_image


@dataclass
class TFDatasetBundle:
    dataset: tf.data.Dataset
    steps_per_epoch: int
    sample_count: int


def build_tf_dataset(
    split: DatasetSplit,
    image_size: Tuple[int, int],
    batch_size: int,
    shuffle: bool,
    augment: bool,
    seed: int,
) -> TFDatasetBundle:
    """Create a tf.data.Dataset pipeline for a dataset split."""
    paths = [str(path) for path in split.files]
    labels = split.labels.astype("int32")

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)

    dataset = dataset.map(
        lambda p, y: load_image(p, y, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    if augment:
        augmenter = build_data_augmentation(seed=seed)
        dataset = dataset.map(
            lambda x, y: apply_augmentation(x, y, augmenter),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    dataset = dataset.map(
        apply_preprocessing,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    steps = math.ceil(len(paths) / batch_size)

    return TFDatasetBundle(dataset=dataset, steps_per_epoch=steps, sample_count=len(paths))

