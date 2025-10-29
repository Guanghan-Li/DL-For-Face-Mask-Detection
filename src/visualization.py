from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import seaborn as sns
from PIL import Image
from plotly.subplots import make_subplots
from sklearn.metrics import auc, precision_recall_curve, roc_curve


def set_plot_style(config) -> None:
    """Apply global plotting style based on configuration."""
    sns.set_theme(style=config.style, palette=config.color_palette)
    plt.rcParams.update(
        {
            "axes.titlesize": "medium",
            "axes.labelsize": "medium",
            "figure.autolayout": True,
        }
    )


def save_figure(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    """Persist a matplotlib figure and release resources."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_training_history_enhanced(
    history_df: pd.DataFrame,
    best_epoch: int,
    lr_series: Iterable[Optional[float]],
    out_path: Path,
    config,
) -> None:
    """Visualize loss/accuracy curves with shading and best-epoch markers."""
    epochs = history_df["epoch"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    train_loss = history_df.get("loss")
    val_loss = history_df.get("val_loss")
    if train_loss is not None and val_loss is not None:
        axes[0].plot(epochs, train_loss, label="Train Loss")
        axes[0].plot(epochs, val_loss, label="Validation Loss")
        axes[0].fill_between(epochs, train_loss, val_loss, alpha=0.2)
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        best_epoch_value = epochs[best_epoch]
        axes[0].axvline(best_epoch_value, color="red", linestyle="--", label="Best Epoch")
        axes[0].legend()

    train_acc = history_df.get("accuracy")
    val_acc = history_df.get("val_accuracy")
    if train_acc is not None and val_acc is not None:
        axes[1].plot(epochs, train_acc, label="Train Accuracy")
        axes[1].plot(epochs, val_acc, label="Validation Accuracy")
        axes[1].fill_between(epochs, train_acc, val_acc, alpha=0.2)
        axes[1].set_ylabel("Accuracy")
        best_epoch_value = epochs[best_epoch]
        axes[1].axvline(best_epoch_value, color="red", linestyle="--", label="Best Epoch")
        axes[1].legend()

    lr_values = [value for value in lr_series if value is not None]
    if lr_values:
        ax_lr = axes[1].twinx()
        lr_epochs = epochs[: len(lr_values)]
        ax_lr.plot(lr_epochs, lr_values, color="grey", linestyle=":", label="Learning Rate")
        ax_lr.set_ylabel("Learning Rate")
        ax_lr.legend(loc="lower right")

    axes[1].set_xlabel("Epoch")
    save_figure(fig, out_path, config.dpi)


def plot_confusion_matrix_enhanced(
    cm: np.ndarray,
    class_names: List[str],
    test_accuracy: float,
    out_path: Path,
    normalize: bool,
    config,
) -> None:
    """Plot a confusion matrix with optional normalization and accuracy title."""
    matrix = cm.astype(float)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True).clip(min=1.0)
        matrix = matrix / row_sums
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (Accuracy: {test_accuracy:.2%})")
    save_figure(fig, out_path, config.dpi)


def plot_per_class_metrics(
    per_class_df: pd.DataFrame,
    class_names: List[str],
    out_path: Path,
    config,
) -> None:
    """Render grouped bars for per-class precision/recall/F1 metrics."""
    metrics_df = per_class_df.set_index("class").loc[class_names].reset_index()
    melt_df = metrics_df.melt(
        id_vars="class",
        value_vars=["precision", "recall", "f1-score"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=melt_df, x="class", y="value", hue="metric", ax=ax)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.legend(title="")
    ax.set_title("Per-Class Metrics")
    save_figure(fig, out_path, config.dpi)


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, out_path: Path, config) -> None:
    """Generate ROC curve for binary classification."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.set_title("Receiver Operating Characteristic")
    save_figure(fig, out_path, config.dpi)


def plot_pr_curve(y_true: np.ndarray, y_score: np.ndarray, out_path: Path, config) -> None:
    """Generate Precision-Recall curve for binary classification."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.set_title("Precision-Recall Curve")
    save_figure(fig, out_path, config.dpi)


def plot_learning_rate_schedule(
    lr_series: Iterable[Optional[float]],
    out_path: Path,
    config,
) -> None:
    """Plot the learning rate evolution over epochs."""
    values = [value for value in lr_series if value is not None]
    if not values:
        return
    epochs = np.arange(1, len(values) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, values, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    save_figure(fig, out_path, config.dpi)


def plot_class_distribution(
    split_counts: Dict[str, Dict[str, int]],
    out_path: Path,
    config,
) -> None:
    """Visualize class distribution across dataset splits."""
    records: List[Dict[str, object]] = []
    for split_name, counts in split_counts.items():
        for class_name, count in counts.items():
            records.append({"split": split_name, "class": class_name, "count": count})
    if not records:
        return
    dist_df = pd.DataFrame.from_records(records)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=dist_df, x="class", y="count", hue="split", ax=ax)
    ax.set_title("Class Distribution per Split")
    save_figure(fig, out_path, config.dpi)


def plot_sample_predictions(
    samples: List[Dict[str, object]],
    out_path: Path,
    config,
) -> None:
    """Create a grid of sample predictions with color-coded correctness."""
    if not samples:
        return
    count = len(samples)
    cols = int(math.ceil(math.sqrt(count)))
    rows = int(math.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    for idx, sample in enumerate(samples):
        row, col = divmod(idx, cols)
        ax = axes[row, col]
        path = Path(sample["image_path"])
        if not path.exists():
            continue
        image = Image.open(path).convert("RGB")
        ax.imshow(image)
        title = f"T: {sample['true_label']}\nP: {sample['pred_label']} ({sample['confidence']:.2f})"
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        color = "green" if sample.get("is_correct") else "red"
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)

    save_figure(fig, out_path, config.dpi)


def render_interactive_dashboard(
    history_csv: Path,
    cm_csv: Path,
    per_cls_csv: Path,
    out_html: Path,
) -> None:
    """Render an interactive dashboard with Plotly and write to HTML."""
    history_df = pd.read_csv(history_csv) if history_csv.exists() else None
    cm_df = pd.read_csv(cm_csv, index_col=0) if cm_csv.exists() else None
    per_cls_df = pd.read_csv(per_cls_csv) if per_cls_csv.exists() else None

    rows, cols = 2, 2
    specs = [[{"type": "xy"}, {"type": "heatmap"}], [{"type": "bar"}, {"type": "table"}]]
    titles = ["Accuracy & Loss", "Confusion Matrix", "Per-Class Metrics", "Top Metrics"]
    fig = make_subplots(rows=rows, cols=cols, specs=specs, subplot_titles=titles)

    if history_df is not None and {"epoch", "accuracy", "val_accuracy", "loss", "val_loss"}.issubset(history_df.columns):
        fig.add_trace(
            go.Scatter(x=history_df["epoch"], y=history_df["accuracy"], name="Train Accuracy"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=history_df["epoch"], y=history_df["val_accuracy"], name="Val Accuracy"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=history_df["epoch"], y=history_df["loss"], name="Train Loss", yaxis="y3"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=history_df["epoch"], y=history_df["val_loss"], name="Val Loss", yaxis="y3"),
            row=1,
            col=1,
        )
        fig.update_layout(
            yaxis=dict(title="Accuracy"),
            yaxis3=dict(title="Loss", overlaying="y", side="right"),
        )

    if cm_df is not None and not cm_df.empty:
        fig.add_trace(
            go.Heatmap(z=cm_df.to_numpy(), x=list(cm_df.columns), y=list(cm_df.index), coloraxis="coloraxis"),
            row=1,
            col=2,
        )
        fig.update_layout(coloraxis=dict(colorscale="Blues"))

    if per_cls_df is not None and not per_cls_df.empty:
        metric_cols = [col for col in ["precision", "recall", "f1-score"] if col in per_cls_df.columns]
        if metric_cols:
            melt_df = per_cls_df.melt(id_vars="class", value_vars=metric_cols, var_name="metric", value_name="value")
            bar_fig = px.bar(melt_df, x="class", y="value", color="metric", barmode="group")
            for trace in bar_fig.data:
                fig.add_trace(trace, row=2, col=1)
        table_columns = [col for col in per_cls_df.columns if col != "support"]
        fig.add_trace(
            go.Table(
                header=dict(values=table_columns, fill_color="#262626", font=dict(color="white")),
                cells=dict(values=[per_cls_df[col] for col in table_columns]),
            ),
            row=2,
            col=2,
        )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(fig, file=str(out_html), auto_open=False, include_plotlyjs="cdn")

