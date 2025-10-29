from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data import collect_image_files, prepare_dataset_bundle, split_datasets


@pytest.fixture()
def logger() -> logging.Logger:
    log = logging.getLogger("test_logger")
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        log.addHandler(logging.StreamHandler())
    return log


def _create_dummy_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.random.randint(0, 255, size=(32, 32, 3), dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_collect_image_files_filters_non_images(tmp_path: Path, logger: logging.Logger) -> None:
    with_mask = tmp_path / "with_mask"
    without_mask = tmp_path / "without_mask"
    _create_dummy_image(with_mask / "sample1.png")
    _create_dummy_image(with_mask / "sample2.jpg")
    _create_dummy_image(without_mask / "sample3.png")
    (with_mask / "notes.txt").write_text("not an image", encoding="utf-8")

    files, labels, class_to_index, class_names = collect_image_files(
        data_root=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        logger=logger,
    )

    assert len(files) == 3
    assert len(labels) == 3
    assert set(class_names) == {"with_mask", "without_mask"}
    assert class_to_index["without_mask"] == 0
    assert class_to_index["with_mask"] == 1


def test_split_datasets_preserves_distribution(logger: logging.Logger) -> None:
    files = [Path(f"/tmp/img_{i}.png") for i in range(20)]
    labels = [0] * 10 + [1] * 10
    train, val, test = split_datasets(
        files=files,
        labels=labels,
        validation_split=0.2,
        test_split=0.2,
        seed=123,
    )

    assert len(train.files) == 12
    assert len(val.files) == 4
    assert len(test.files) == 4
    assert set(np.unique(train.labels)) == {0, 1}
    assert set(np.unique(val.labels)) == {0, 1}
    assert set(np.unique(test.labels)) == {0, 1}


def test_prepare_dataset_bundle_generates_artifacts(tmp_path: Path, logger: logging.Logger) -> None:
    with_mask = tmp_path / "with_mask"
    without_mask = tmp_path / "without_mask"
    for idx in range(5):
        _create_dummy_image(with_mask / f"mask_{idx}.png")
        _create_dummy_image(without_mask / f"nomask_{idx}.png")

    bundle = prepare_dataset_bundle(
        data_root=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        validation_split=0.2,
        test_split=0.2,
        seed=42,
        logger=logger,
    )

    assert bundle.train.files
    assert bundle.validation.files
    assert bundle.test.files
    assert (tmp_path / "artifacts").exists()
