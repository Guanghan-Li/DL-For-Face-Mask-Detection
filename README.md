# DL-For-Face-Mask-Detection

## Overview
Modular face-mask detection pipeline powered by MobileNetV2 transfer learning. Data ingestion, preprocessing, training, evaluation, and inference now live under the `src/` package with reproducible configs, structured logging, and run artifacts.

## Dataset
- Expected layout: `archive/data/<class_name>` with subfolders such as `with_mask` and `without_mask`.
- Only files with extensions `.png .jpg .jpeg .bmp .gif .tif .tiff` are loaded; others are ignored automatically.
- Adjust `data.root` in `configs/base.yaml` if your dataset directory changes. The default points to `archive/data`.

## Environment Setup
1. Create and activate a Python 3.9+ virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. GPU acceleration is recommended; install the GPU-enabled TensorFlow build where available.

## Training
- Kick off a run with either command:
  ```bash
  python -m src.train --config configs/base.yaml
  # or
  python main.py train --config configs/base.yaml
  ```
- Artifacts (logs, metrics, plots, checkpoints) are written to `artifacts/<timestamp>-<run-name>/`.
- The best-performing model for deployment is also exported to `models/best_model.keras`.
- Data augmentation mirrors the prior ImageDataGenerator setup (flips, rotations, translations, zoom, brightness) but now runs inside the `tf.data` pipeline.

## Evaluation & Outputs
- `artifacts/<timestamp>/metrics/metrics.json`: loss, accuracy, precision, recall, F1, plus ROC/PR AUC (binary runs).
- `artifacts/<timestamp>/metrics/classification_report.txt`: full precision/recall table.
- `artifacts/<timestamp>/metrics/epoch_metrics.csv`: per-epoch history for reuse in dashboards.
- `artifacts/<timestamp>/metadata.json`: frozen config, label mappings, and artifact index.

## Visualization & Metrics
- Static plots in `artifacts/<timestamp>/plots/`:
  - `training_history_enhanced.png`, `confusion_matrix_enhanced.png`, `per_class_metrics.png`
  - Optional: `learning_rate_schedule.png`, `class_distribution.png`, `roc_curve.png`, `precision_recall_curve.png`, `sample_predictions.png`
- Interactive Plotly dashboard: `artifacts/<timestamp>/plots/metrics_dashboard.html`
- TensorBoard logs: `artifacts/<timestamp>/tensorboard/`
- Normalized confusion matrix and per-class stats CSVs live in `artifacts/<timestamp>/metrics/`

Open the artifacts with:
```bash
python main.py train --config configs/base.yaml --run-name viz-upgrade
tensorboard --logdir artifacts/<timestamp>-viz-upgrade/tensorboard
```
- CSVs back the Plotly dashboard and can feed downstream analytics notebooks.

## Prediction CLI
Use the dedicated predictor to score individual images:
```bash
python -m src.predict \
  --model-path models/best_model.keras \
  --metadata artifacts/<timestamp>-run/metadata.json \
  --image path/to/example.jpg
```
Add `--output prediction.json` to persist the result.

## Tests & Quality Gates
- Run the lightweight data and pipeline checks with `pytest`.
- Future additions (e.g., linting, continuous integration) can hook into these foundations.

## Tips
- Seeds, splits, and preprocessing are centralized in the config; tweak `configs/base.yaml` for experiments.
- Track multiple experiments by supplying `--run-name` to stamp artifacts.
