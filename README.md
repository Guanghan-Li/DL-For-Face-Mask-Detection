# DL-For-Face-Mask-Detection

## Overview
Face-mask detection pipeline built with transfer learning on MobileNetV2. The script in `main.py` loads face images with and without masks, augments the training set, fine-tunes a lightweight convolutional neural network, and reports accuracy, precision, recall, F1, and a confusion matrix.

## Dataset
- Expected folder layout: `archive/data/with_mask` and `archive/data/without_mask` containing RGB images.
- Update the absolute paths defined near the top of `main.py` if your dataset lives elsewhere.
- Current dataset size: 3,725 mask images and 3,828 non-mask images.

## Environment Setup
1. Create and activate a Python 3.9+ virtual environment.
2. Install dependencies:
   ```bash
   pip install numpy matplotlib seaborn pillow scikit-learn tensorflow
   ```
3. Ensure the `models/` directory exists for saving checkpoints and artifacts.

## Training
- Run `python main.py` from the project root to start training.
- Data augmentation covers rotation, shifts, flips, zoom, brightness jitter, and rescaling.
- Training uses MobileNetV2 (frozen ImageNet weights), global average pooling, a 128-unit dense layer, dropout (0.5), and a sigmoid output.
- Callbacks include `ModelCheckpoint` (best weights to `models/best_model.h5`), `EarlyStopping`, and `ReduceLROnPlateau`.

## Evaluation & Outputs
- After training, the script evaluates the held-out test set and prints classification metrics plus a confusion matrix.
- Plots are saved to `models/confusion_matrix.png` and `models/training_history.png`.
- The final model is exported to `models/final_model.h5`.

## Interactive Prediction
Once training completes, the CLI prompts for an image path. Provide any image file, and the model will output whether a mask is detected along with the confidence percentage.

## Tips
- For reproducibility, data splits use `random_state=2`; adjust if you need different splits.
- Use a GPU-enabled TensorFlow build for faster training.
