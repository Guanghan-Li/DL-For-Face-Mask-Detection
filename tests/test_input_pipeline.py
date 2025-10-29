from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

tf = pytest.importorskip("tensorflow")

from src.data import DatasetSplit
from src.input_pipeline import build_tf_dataset


def _create_dummy_image(path, size=(16, 16)):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.random.randint(0, 255, size=(*size, 3), dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_build_tf_dataset_shapes(tmp_path):
    img_paths = []
    labels = []
    for idx in range(3):
        path = tmp_path / f"image_{idx}.png"
        _create_dummy_image(path)
        img_paths.append(path)
        labels.append(idx % 2)

    split = DatasetSplit(files=img_paths, labels=np.array(labels))
    dataset_bundle = build_tf_dataset(
        split=split,
        image_size=(128, 128),
        batch_size=2,
        shuffle=False,
        augment=False,
        seed=123,
    )

    batch_images, batch_labels = next(iter(dataset_bundle.dataset))
    assert batch_images.shape[1:] == (128, 128, 3)
    assert tf.reduce_max(batch_images).numpy() <= 1.0
    assert tf.reduce_min(batch_images).numpy() >= -1.0
    assert batch_labels.shape[0] == 2
