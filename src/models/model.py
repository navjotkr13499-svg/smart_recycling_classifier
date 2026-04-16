"""
Model architecture for Smart Recycling Classifier
Using MobileNetV2 with Transfer Learning
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from config import *


# ============================================================
# 🧠 BUILD MODEL
# ============================================================
def build_model(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    """
    Build MobileNetV2 transfer learning model

    Args:
        input_shape : tuple (H, W, C) — default (224, 224, 3)
        num_classes : int             — default 6

    Returns:
        model : compiled Keras model
    """

    # ── Step 1: Load pretrained base ──────────────────────────
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,       # remove ImageNet classifier head
        weights='imagenet'       # use pretrained weights
    )

    # ── Step 2: Freeze base layers ────────────────────────────
    base_model.trainable = False
    print(f"✅ Base model loaded  : {base_model.name}")
    print(f"   Total layers       : {len(base_model.layers)}")
    print(f"   Trainable params   : {base_model.count_params():,} (frozen)")

    # ── Step 3: Build custom head ─────────────────────────────
    inputs = tf.keras.Input(shape=input_shape, name="input_image")

    # Pass through frozen base
    x = base_model(inputs, training=False)

    # Global Average Pooling (reduces spatial dims → single vector)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Batch Normalization for stable training
    x = layers.BatchNormalization(name="batch_norm")(x)

    # Dense layer with L2 regularization
    x = layers.Dense(
        DENSE_UNITS,
        activation='relu',
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_head"
    )(x)

    # Dropout to reduce overfitting
    x = layers.Dropout(DROPOUT_RATE, name="dropout")(x)

    # Final classification layer
    outputs = layers.Dense(
        num_classes,
        activation='softmax',
        name="output_predictions"
    )(x)

    # ── Step 4: Assemble model ────────────────────────────────
    model = models.Model(inputs=inputs, outputs=outputs, name="RecycleNet")

    # ── Step 5: Compile ───────────────────────────────────────
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')
        ]
    )

    return model


# ============================================================
# 🔓 FINE-TUNING — Unfreeze top N layers
# ============================================================
def unfreeze_model(model, num_layers=FINE_TUNE_LAYERS, learning_rate=1e-5):
    """
    Unfreeze the top N layers of the base model for fine-tuning.

    Args:
        model        : trained Keras model (after initial training)
        num_layers   : how many layers from the end to unfreeze
        learning_rate: lower LR for fine-tuning (avoid catastrophic forgetting)

    Returns:
        model : recompiled model ready for fine-tuning
    """
    # Get the base model (first layer after input)
    base_model = model.layers[1]

    # Unfreeze the whole base first
    base_model.trainable = True

    # Re-freeze everything except last `num_layers`
    for layer in base_model.layers[:-num_layers]:
        layer.trainable = False

    # Count trainable params after unfreeze
    trainable = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )

    print(f"\n🔓 Fine-tuning enabled!")
    print(f"   Unfrozen layers    : last {num_layers} of base model")
    print(f"   Trainable params   : {trainable:,}")
    print(f"   Learning rate      : {learning_rate}")

    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')
        ]
    )

    return model


# ============================================================
# 💾 SAVE / LOAD
# ============================================================
def save_model(model, filename="recycle_net.keras"):
    """Save model to MODELS_DIR"""
    save_path = MODELS_DIR / filename
    model.save(save_path)
    print(f"💾 Model saved → {save_path}")
    return save_path


def load_model(filename="recycle_net.keras"):
    """Load model from MODELS_DIR"""
    load_path = MODELS_DIR / filename
    model = tf.keras.models.load_model(load_path)
    print(f"📂 Model loaded ← {load_path}")
    return model


# ============================================================
# 🔍 SUMMARY UTIL
# ============================================================
def print_model_summary(model):
    """Print detailed model summary"""
    print("\n" + "=" * 60)
    print("🧠 MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)
    model.summary()
    print("=" * 60)

    total     = model.count_params()
    trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
    frozen    = total - trainable

    print(f"\n📊 Parameter Breakdown:")
    print(f"   Total params      : {total:>12,}")
    print(f"   Trainable params  : {trainable:>12,}")
    print(f"   Frozen params     : {frozen:>12,}")
    print("=" * 60)


# ============================================================
# ▶️ QUICK TEST
# ============================================================
if __name__ == "__main__":
    print("🚀 Building RecycleNet...\n")

    model = build_model()
    print_model_summary(model)

    # Sanity check — run a dummy batch through
    import numpy as np
    dummy = np.random.rand(1, *INPUT_SHAPE).astype("float32")
    preds = model.predict(dummy, verbose=0)

    print(f"\n✅ Forward pass OK!")
    print(f"   Input shape  : {dummy.shape}")
    print(f"   Output shape : {preds.shape}")
    print(f"   Predictions  : {dict(zip(CLASS_NAMES, preds[0].round(4)))}")
