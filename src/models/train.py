"""
Training pipeline for Smart Recycling Classifier
Includes: callbacks, phase-1 training, fine-tuning, plotting
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping,
    ReduceLROnPlateau, CSVLogger
)
from config import *
from src.models.model import build_model, unfreeze_model, save_model, print_model_summary
from src.data.augmentation import create_data_generators


# ============================================================
# 📞 CALLBACKS
# ============================================================
def build_callbacks(phase="phase1"):
    """
    Build training callbacks

    Args:
        phase : "phase1" (frozen) or "phase2" (fine-tune)

    Returns:
        list of Keras callbacks
    """
    checkpoint_path = MODELS_DIR / f"best_model_{phase}.keras"
    csv_path        = RESULTS_DIR / "metrics" / f"training_log_{phase}.csv"

    # Create dirs
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [

        # ── Save best model ───────────────────────────────────
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),

        # ── Stop early if no improvement ──────────────────────
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),

        # ── Reduce LR on plateau ──────────────────────────────
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),

        # ── CSV training log ──────────────────────────────────
        CSVLogger(
            filename=str(csv_path),
            append=False
        ),
    ]

    print(f"✅ Callbacks ready for [{phase}]")
    print(f"   Best model  → {checkpoint_path}")
    print(f"   CSV log     → {csv_path}")
    return callbacks


# ============================================================
# 📈 PLOT TRAINING HISTORY
# ============================================================
def plot_history(history, phase="phase1"):
    """
    Plot accuracy and loss curves

    Args:
        history : Keras History object
        phase   : label for saving the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training History — {phase}', fontsize=16, fontweight='bold')

    epochs = range(1, len(history.history['accuracy']) + 1)

    # ── Accuracy ─────────────────────────────────────────────
    axes[0].plot(epochs, history.history['accuracy'],     'b-o', label='Train Accuracy',  markersize=4)
    axes[0].plot(epochs, history.history['val_accuracy'], 'r-o', label='Val Accuracy',    markersize=4)
    axes[0].set_title('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Loss ─────────────────────────────────────────────────
    axes[1].plot(epochs, history.history['loss'],     'b-o', label='Train Loss',  markersize=4)
    axes[1].plot(epochs, history.history['val_loss'], 'r-o', label='Val Loss',    markersize=4)
    axes[1].set_title('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    save_path = RESULTS_DIR / "plots" / f"training_history_{phase}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plot saved → {save_path}")
    plt.close()


# ============================================================
# 🏋️ PHASE 1 — Train with frozen base
# ============================================================
def train_phase1(model, train_gen, val_gen, epochs=30):
    """
    Phase 1: Train only the custom head (base frozen)

    Args:
        model     : compiled Keras model
        train_gen : training data generator
        val_gen   : validation data generator
        epochs    : max epochs

    Returns:
        model, history
    """
    print("\n" + "=" * 60)
    print("🏋️  PHASE 1 — Training custom head (base frozen)")
    print("=" * 60)

    callbacks = build_callbacks(phase="phase1")

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # Plot & save
    plot_history(history, phase="phase1")
    save_model(model, "recycle_net_phase1.keras")

    best_val_acc = max(history.history['val_accuracy'])
    print(f"\n✅ Phase 1 complete!")
    print(f"   Best Val Accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

    return model, history


# ============================================================
# 🔓 PHASE 2 — Fine-tune unfrozen layers
# ============================================================
def train_phase2(model, train_gen, val_gen, epochs=20):
    """
    Phase 2: Fine-tune top layers of base model

    Args:
        model     : model from phase 1
        train_gen : training data generator
        val_gen   : validation data generator
        epochs    : max epochs

    Returns:
        model, history
    """
    print("\n" + "=" * 60)
    print("🔓 PHASE 2 — Fine-tuning top layers")
    print("=" * 60)

    # Unfreeze top layers with lower LR
    model = unfreeze_model(model, num_layers=FINE_TUNE_LAYERS, learning_rate=1e-5)

    callbacks = build_callbacks(phase="phase2")

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # Plot & save
    plot_history(history, phase="phase2")
    save_model(model, "recycle_net_final.keras")

    best_val_acc = max(history.history['val_accuracy'])
    print(f"\n✅ Phase 2 complete!")
    print(f"   Best Val Accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

    return model, history


# ============================================================
# 🚀 FULL TRAINING PIPELINE
# ============================================================
def run_training():
    """
    Full end-to-end training pipeline:
    Phase 1 → Phase 2 (fine-tune) → Save final model
    """
    print("\n" + "=" * 60)
    print("🚀 SMART RECYCLING CLASSIFIER — TRAINING PIPELINE")
    print("=" * 60)

    # ── Step 1: Data ──────────────────────────────────────────
    print("\n📂 Loading data generators...")
    train_gen, val_gen, test_gen = create_data_generators(batch_size=16)
    print(f"   Train : {train_gen.samples} images")
    print(f"   Val   : {val_gen.samples} images")
    print(f"   Test  : {test_gen.samples} images")

    # ── Step 2: Build model ───────────────────────────────────
    print("\n🧠 Building model...")
    model = build_model()
    print_model_summary(model)

    # ── Step 3: Phase 1 training ──────────────────────────────
    model, history1 = train_phase1(model, train_gen, val_gen, epochs=30)

    # ── Step 4: Phase 2 fine-tuning ───────────────────────────
    model, history2 = train_phase2(model, train_gen, val_gen, epochs=20)

    # ── Step 5: Final summary ─────────────────────────────────
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print(f"   Phase 1 best val acc : {max(history1.history['val_accuracy'])*100:.2f}%")
    print(f"   Phase 2 best val acc : {max(history2.history['val_accuracy'])*100:.2f}%")
    print(f"   Final model saved    : {MODELS_DIR / 'recycle_net_final.keras'}")
    print(f"   Plots saved          : {RESULTS_DIR / 'plots'}")
    print("=" * 60)

    return model, history1, history2


# ============================================================
# ▶️ ENTRY POINT
# ============================================================
if __name__ == "__main__":
    model, history1, history2 = run_training()
