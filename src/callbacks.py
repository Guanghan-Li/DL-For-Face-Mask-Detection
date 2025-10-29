from __future__ import annotations

from typing import Optional

import tensorflow as tf


class LearningRateLogger(tf.keras.callbacks.Callback):
    """Track the learning rate applied at the end of every epoch."""

    def __init__(self) -> None:
        super().__init__()
        self.learning_rates: list[Optional[float]] = []

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:  # pragma: no cover - signature enforced by Keras
        del epoch, logs
        optimizer = getattr(self.model, "optimizer", None)  # type: ignore[attr-defined]
        lr = None
        if optimizer is not None:
            lr = getattr(optimizer, "lr", None) or getattr(optimizer, "learning_rate", None)
        lr_value: Optional[float] = None
        if lr is not None:
            try:
                lr_value = float(tf.keras.backend.get_value(lr))
            except Exception:  # pragma: no cover - fallback path
                if hasattr(lr, "numpy"):
                    lr_value = float(lr.numpy())
        self.learning_rates.append(lr_value)

