from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


@dataclass
class DataConfig:
    root: Path
    image_size: Tuple[int, int]
    validation_split: float
    test_split: float
    batch_size: int


@dataclass
class TrainingConfig:
    epochs: int
    learning_rate: float
    dense_units: int
    dropout_rate: float
    early_stopping_patience: int
    reduce_lr_patience: int
    reduce_lr_factor: float
    min_lr: float


@dataclass
class ArtifactsConfig:
    root: Path
    model_registry: Path


@dataclass
class Config:
    seed: int
    data: DataConfig
    training: TrainingConfig
    artifacts: ArtifactsConfig


def load_config(path: Path) -> Config:
    """Load configuration from YAML into strongly typed dataclasses."""
    with path.open("r", encoding="utf-8") as handle:
        raw: Dict[str, Any] = yaml.safe_load(handle)

    base_dir = path.parent.resolve()

    data_cfg = raw["data"]
    data = DataConfig(
        root=(base_dir / data_cfg["root"]).resolve(),
        image_size=tuple(data_cfg["image_size"]),
        validation_split=float(data_cfg["validation_split"]),
        test_split=float(data_cfg["test_split"]),
        batch_size=int(data_cfg["batch_size"]),
    )

    training_cfg = raw["training"]
    training = TrainingConfig(
        epochs=int(training_cfg["epochs"]),
        learning_rate=float(training_cfg["learning_rate"]),
        dense_units=int(training_cfg["dense_units"]),
        dropout_rate=float(training_cfg["dropout_rate"]),
        early_stopping_patience=int(training_cfg["early_stopping_patience"]),
        reduce_lr_patience=int(training_cfg["reduce_lr_patience"]),
        reduce_lr_factor=float(training_cfg["reduce_lr_factor"]),
        min_lr=float(training_cfg["min_lr"]),
    )

    artifacts_cfg = raw["artifacts"]
    artifacts = ArtifactsConfig(
        root=(base_dir / artifacts_cfg["root"]).resolve(),
        model_registry=(base_dir / artifacts_cfg["model_registry"]).resolve(),
    )

    return Config(
        seed=int(raw["seed"]),
        data=data,
        training=training,
        artifacts=artifacts,
    )


def config_to_dict(config: Config) -> Dict[str, Any]:
    """Convert config dataclasses back into primitive dictionary for logging/serialization."""
    raw = dataclasses.asdict(config)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {key: _convert(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [_convert(item) for item in obj]
        return obj

    return _convert(raw)
