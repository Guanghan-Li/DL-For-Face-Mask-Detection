import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Paths to dataset
with_mask_path = '/Users/garrick/codes/Machine Learning Projects/Facemask Detection/archive/data/with_mask'
with_mask_files = os.listdir(with_mask_path)

without_mask_path = '/Users/garrick/codes/Machine Learning Projects/Facemask Detection/archive/data/without_mask'
without_mask_files = os.listdir(without_mask_path)

# Create labels
with_mask_labels = [1] * 3725
without_mask_labels = [0] * 3828
labels = with_mask_labels + without_mask_labels

# Manual data loading (kept for train_test_split compatibility)
data = []

with_mask_path_full = '/Users/garrick/codes/Machine Learning Projects/Facemask Detection/archive/data/with_mask/'
for img_file in with_mask_files:
    image = Image.open(with_mask_path_full + img_file)
    image = image.resize((128, 128))
    image = image.convert('RGB')
    image = np.array(image)
    data.append(image)

without_mask_path_full = '/Users/garrick/codes/Machine Learning Projects/Facemask Detection/archive/data/without_mask/'
for img_file in without_mask_files:
    image = Image.open(without_mask_path_full + img_file)
    image = image.resize((128, 128))
    image = image.convert('RGB')
    image = np.array(image)
    data.append(image)

# Convert to numpy arrays
x = np.array(data)
y = np.array(labels)

# Train-validation-test split
x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.2, random_state=2)

# Create ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    rescale=1./255
)

# Validation and test generators without augmentation
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators using flow method
batch_size = 32
train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True)
val_generator = val_datagen.flow(x_val, y_val, batch_size=batch_size, shuffle=False)
test_generator = test_datagen.flow(x_test, y_test, batch_size=batch_size, shuffle=False)

# Calculate steps
steps_per_epoch = len(x_train) // batch_size
validation_steps = len(x_val) // batch_size

# Build transfer learning model with MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

# Build model architecture
model = tf.keras.Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile model with binary classification settings
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print model summary
print("\nModel Architecture:")
model.summary()

# Setup training callbacks
checkpoint = ModelCheckpoint(
    filepath='models/best_model.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

callbacks = [checkpoint, early_stopping, reduce_lr]

# Train model
print("\nStarting model training...")
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_generator,
    validation_steps=validation_steps,
    epochs=30,
    callbacks=callbacks,
    verbose=1
)

# Evaluate model on test set
print("\nEvaluating model on test set...")
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f'\nTest Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Generate predictions for detailed metrics
print("\nGenerating predictions for detailed metrics...")
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

# Calculate comprehensive evaluation metrics
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f'\nPrecision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Mask', 'With Mask']))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Mask', 'With Mask'],
            yticklabels=['No Mask', 'With Mask'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png')
plt.show()

# Plot training history - Loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Plot training history - Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('models/training_history.png')
plt.show()

# Save final model
print("\nSaving final model...")
model.save('models/final_model.h5')
print("Final model saved to 'models/final_model.h5'")

# Predictive system
print("\n" + "="*50)
print("PREDICTION SYSTEM")
print("="*50)
input_image_path = input("Path of the image to be predicted: ")

try:
    input_image = Image.open(input_image_path)
    input_image_resized = input_image.resize((128, 128))
    input_image_array = np.array(input_image_resized)
    input_image_scaled = input_image_array / 255.0
    input_image_reshaped = np.reshape(input_image_scaled, [1, 128, 128, 3])
    
    input_prediction = model.predict(input_image_reshaped, verbose=0)
    confidence = input_prediction[0][0]
    
    print(f"\nPrediction probability: {confidence:.4f}")
    
    if confidence > 0.5:
        print(f"Result: Wearing mask (Confidence: {confidence*100:.2f}%)")
    else:
        print(f"Result: No mask (Confidence: {(1-confidence)*100:.2f}%)")
        
except Exception as e:
    print(f"Error processing image: {e}")

print("\nTraining complete!")
