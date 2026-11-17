
import os
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    RandomFlip,
    RandomRotation,
    RandomZoom
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from collections import Counter

# Dataset
DATASET_PATH = "dataset/"

IMG_SIZE = (128, 128)
BATCH_SIZE = 4
EPOCHS = 50

# Count classes
num_classes = len([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
])

class_mode = "binary" if num_classes == 2 else "categorical"

# Load images
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=class_mode,
    subset="training",
    seed=42
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=class_mode,
    subset="validation",
    shuffle=False,
    seed=42
)

# Show class distribution
print("Class names:", train_data.class_indices)
print("Training class counts:",
      Counter(train_data.classes))
print("Validation class counts:",
      Counter(val_data.classes))

# Calculate class weights
counts = Counter(train_data.classes)
total = len(train_data.classes)

class_weights = {
    class_id: total / (num_classes * count)
    for class_id, count in counts.items()
}

# Improved CNN
model = Sequential([
    Input(shape=(128, 128, 3)),

    # Data augmentation (training only)
    RandomFlip("horizontal"),
    RandomRotation(0.08),
    RandomZoom(0.1),

    # First block
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(2, 2),

    # Second block
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(2, 2),

    # Third block
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    MaxPooling2D(2, 2),

    # Classification
    GlobalAveragePooling2D(),

    Dense(64, activation="relu"),
    Dropout(0.4),

    Dense(
        1 if class_mode == "binary" else num_classes,
        activation="sigmoid" if class_mode == "binary"
        else "softmax"
    )
])

# Compile
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0003
    ),
    loss="binary_crossentropy" if class_mode == "binary"
    else "categorical_crossentropy",
    metrics=["accuracy"]
)

# Callbacks
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=0.00001,
    verbose=1
)

# Train
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=[early_stopping, reduce_lr],
    class_weight=class_weights
)

# Evaluate
val_loss, val_accuracy = model.evaluate(val_data)

print("Validation accuracy:", round(val_accuracy * 100, 2), "%")

# Save
model.save("improved_cats_dogs.keras")
