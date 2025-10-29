from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}


@dataclass
class DatasetSplit:
    files: List[Path]
    labels: np.ndarray


@dataclass
class DatasetBundle:
    class_names: List[str]
    class_to_index: Dict[str, int]
    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit


def collect_image_files(
    data_root: Path,
    artifacts_dir: Path,
    logger: logging.Logger,
) -> Tuple[List[Path], List[int], Dict[str, int], List[str]]:
    """Scan the dataset directory, filter non-image files, and log corrupt samples."""
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_root}")

    preferred_order = {"without_mask": 0, "with_mask": 1}
    class_dirs = sorted(
        [item for item in data_root.iterdir() if item.is_dir()],
        key=lambda item: (preferred_order.get(item.name, 100), item.name),
    )
    if not class_dirs:
        raise ValueError(f"No class folders found under {data_root}")

    class_names = [item.name for item in class_dirs]
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    valid_paths: List[Path] = []
    labels: List[int] = []
    failures: List[Tuple[str, str]] = []

    for class_dir in class_dirs:
        label_index = class_to_index[class_dir.name]
        logger.info("Scanning %s", class_dir)
        for file_path in class_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except (UnidentifiedImageError, OSError) as exc:
                logger.warning("Corrupt image skipped: %s (%s)", file_path, exc)
                failures.append((str(file_path), str(exc)))
                continue
            valid_paths.append(file_path.resolve())
            labels.append(label_index)

    if failures:
        audit_path = artifacts_dir / "load_failures.csv"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "error"])
            writer.writerows(failures)
        logger.info("Wrote %d corrupt file records to %s", len(failures), audit_path)

    logger.info(
        "Discovered %d valid images across %d classes",
        len(valid_paths),
        len(class_names),
    )
    return valid_paths, labels, class_to_index, class_names


def split_datasets(
    files: Sequence[Path],
    labels: Sequence[int],
    validation_split: float,
    test_split: float,
    seed: int,
) -> Tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
    """Split dataset into train/validation/test with stratification."""
    if not 0 < test_split < 1:
        raise ValueError("test_split must be between 0 and 1.")
    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1.")
    if validation_split + test_split >= 1:
        raise ValueError("validation_split + test_split must be less than 1.")

    files_array = np.array([str(path) for path in files])
    labels_array = np.array(labels)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        files_array,
        labels_array,
        test_size=test_split,
        random_state=seed,
        stratify=labels_array,
    )

    relative_val_split = validation_split / (1 - test_split)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=relative_val_split,
        random_state=seed,
        stratify=y_train_val,
    )

    def _to_split(x: np.ndarray, y: np.ndarray) -> DatasetSplit:
        return DatasetSplit(files=[Path(item) for item in x.tolist()], labels=y)

    return _to_split(x_train, y_train), _to_split(x_val, y_val), _to_split(x_test, y_test)


def prepare_dataset_bundle(
    data_root: Path,
    artifacts_dir: Path,
    validation_split: float,
    test_split: float,
    seed: int,
    logger: logging.Logger,
) -> DatasetBundle:
    """Create stratified dataset splits and ensure bookkeeping artifacts exist."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    files, labels, class_to_index, class_names = collect_image_files(
        data_root=data_root,
        artifacts_dir=artifacts_dir,
        logger=logger,
    )
    train, val, test = split_datasets(
        files=files,
        labels=labels,
        validation_split=validation_split,
        test_split=test_split,
        seed=seed,
    )

    def _summarize(split: DatasetSplit, name: str) -> None:
        unique, counts = np.unique(split.labels, return_counts=True)
        distribution = {class_names[idx]: int(count) for idx, count in zip(unique, counts)}
        logger.info("Split %s has %d samples: %s", name, len(split.files), distribution)

    _summarize(train, "train")
    _summarize(val, "validation")
    _summarize(test, "test")

    return DatasetBundle(
        class_names=class_names,
        class_to_index=class_to_index,
        train=train,
        validation=val,
        test=test,
    )
