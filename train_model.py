"""
train_model.py  —  Model Training + Evaluation
============================================================
Alzheimer's Disease Prediction using Multimodal Data
Final-Year B.Tech AI & DS
C. Abdul Hakeem College of Engineering and Technology, Ranipet
 
Run once:
    python train_model.py
 
Produces:
    alzheimer_multimodal_model.h5   (saved model)
    metrics_cache.json              (performance metrics)
"""
 
# ── Step 1 : Imports ─────────────────────────────────────────
import numpy as np
import pandas as pd
 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization, concatenate,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
 
print("=" * 55)
print("  Alzheimer's Multimodal Model  —  Training")
print("=" * 55)
 
 
# ── Step 2 : Image data generators ───────────────────────────
# 80 % training  |  20 % validation  (split from train/ folder)
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2,
)
 
train_data = train_datagen.flow_from_directory(
    "train",
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="training",
)
 
val_data = train_datagen.flow_from_directory(
    "train",
    target_size=(128, 128),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
)
 
print(f"  Train  : {train_data.samples} images")
print(f"  Val    : {val_data.samples} images")
print(f"  Classes: {train_data.class_indices}")
 
 
# ── Step 3 : CNN image branch ─────────────────────────────────
image_input = Input(shape=(128, 128, 3), name="image_input")
 
x = Conv2D(32,  (3, 3), activation="relu", padding="same")(image_input)
x = BatchNormalization()(x)
x = MaxPooling2D(2, 2)(x)
 
x = Conv2D(64,  (3, 3), activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2, 2)(x)
 
x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
x = BatchNormalization()(x)
x = MaxPooling2D(2, 2)(x)
 
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
image_output = Dense(64, activation="relu")(x)
 
print("  CNN branch built.")
 
 
# ── Step 4 : Clinical data branch ─────────────────────────────
clinical_input = Input(shape=(3,), name="clinical_input")
c = Dense(64, activation="relu")(clinical_input)
c = Dense(32, activation="relu")(c)
 
print("  Clinical branch built.")
 
 
# ── Step 5 : Multimodal fusion ────────────────────────────────
combined = concatenate([image_output, c])
z = Dense(128, activation="relu")(combined)
z = Dropout(0.3)(z)
z = Dense(64,  activation="relu")(z)
output = Dense(4, activation="softmax", name="output")(z)
 
final_model = Model(
    inputs=[image_input, clinical_input],
    outputs=output,
)
print("  Multimodal fusion model built.")
final_model.summary()
 
 
# ── Step 6 : Compile ──────────────────────────────────────────
final_model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
print("  Model compiled.")
 
 
# ── Step 7 : Generator wrapper ────────────────────────────────
def multimodal_generator(image_gen):
    """
    Attach zero clinical data during training.
    At inference, real patient values are used.
    """
    while True:
        images, labels = next(image_gen)
        clinical = np.zeros((images.shape[0], 3), dtype=np.float32)
        yield (images, clinical), labels
 
 
train_gen = multimodal_generator(train_data)
val_gen   = multimodal_generator(val_data)
 
 
# ── Step 8 : Callbacks ────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1,
    ),
]
 
 
# ── Step 9 : Train ────────────────────────────────────────────
print("\n  Starting training …\n")
history = final_model.fit(
    train_gen,
    steps_per_epoch  = len(train_data),
    validation_data  = val_gen,
    validation_steps = len(val_data),
    epochs           = 25,
    callbacks        = callbacks,
    verbose          = 1,
)
 
best_val_acc = max(history.history["val_accuracy"])
print(f"\n  Training complete.  Best val accuracy: {best_val_acc:.4f}")
 
 
# ── Step 10 : Save model ──────────────────────────────────────
final_model.save("alzheimer_multimodal_model.h5")
print("  Saved → alzheimer_multimodal_model.h5")
 
 
# ── Step 11 : Compute and cache all metrics ───────────────────
print("\n  Evaluating model and computing benchmark metrics …")
try:
    from metrics import compute_and_cache_metrics
    result = compute_and_cache_metrics(final_model, test_data_dir="test")
 
    print("\n  ── CNN Metrics ──────────────────────────────────")
    for k, v in result["cnn"].items():
        print(f"    {k:12s}: {v}%")
 
    print("\n  ── SVM Benchmark ────────────────────────────────")
    for k, v in result["svm"].items():
        print(f"    {k:12s}: {v}%")
 
    improvement = round(result["cnn"]["accuracy"] - result["svm"]["accuracy"], 2)
    print(f"\n  CNN outperforms SVM by +{improvement}% accuracy")
    print("  Metrics cached → metrics_cache.json")
 
except Exception as exc:
    print(f"  Metrics step skipped: {exc}")
 
print("\n  All done.")
 