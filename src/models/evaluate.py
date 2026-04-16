"""
Evaluation pipeline for Smart Recycling Classifier
Includes: confusion matrix, classification report, per-class accuracy
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    top_k_accuracy_score
)
from config import *
from src.models.model import load_model
from src.data.augmentation import create_data_generators


# ============================================================
# 🔍 GET PREDICTIONS
# ============================================================
def get_predictions(model, generator):
    """
    Run inference on a generator and return true + predicted labels

    Args:
        model     : trained Keras model
        generator : Keras data generator (test/val)

    Returns:
        y_true  : true class indices
        y_pred  : predicted class indices
        y_proba : raw softmax probabilities
    """
    print(f"\n🔍 Running inference on {generator.samples} images...")

    # Reset generator to start
    generator.reset()

    y_proba = model.predict(generator, verbose=1)
    y_pred  = np.argmax(y_proba, axis=1)
    y_true  = generator.classes

    print(f"✅ Inference complete!")
    return y_true, y_pred, y_proba


# ============================================================
# 📊 CONFUSION MATRIX
# ============================================================
def plot_confusion_matrix(y_true, y_pred, normalize=True, phase="test"):
    """
    Plot and save confusion matrix

    Args:
        y_true    : true labels
        y_pred    : predicted labels
        normalize : normalize by row (recall per class)
        phase     : label for filename
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_plot = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt, title_tag = '.2f', '(Normalized)'
    else:
        cm_plot = cm
        fmt, title_tag = 'd', '(Raw Counts)'

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Confusion Matrix — Smart Recycling Classifier',
                 fontsize=16, fontweight='bold')

    for ax, (data, label, f) in zip(
        axes,
        [(cm_plot, title_tag, fmt),
         (cm.astype('float') / cm.sum(axis=1, keepdims=True)
          if not normalize else cm, 'Raw Counts' if normalize else 'Normalized', 'd' if normalize else '.2f')]
    ):
        sns.heatmap(
            data,
            annot=True,
            fmt=f,
            cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            linewidths=0.5,
            ax=ax,
            cbar_kws={'shrink': 0.8}
        )
        ax.set_title(label, fontsize=13)
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()

    save_path = RESULTS_DIR / "plots" / f"confusion_matrix_{phase}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Confusion matrix saved → {save_path}")
    plt.close()

    return cm


# ============================================================
# 📋 CLASSIFICATION REPORT
# ============================================================
def print_classification_report(y_true, y_pred):
    """
    Print and save full classification report

    Args:
        y_true : true labels
        y_pred : predicted labels
    """
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )

    print("\n" + "=" * 60)
    print("📋 CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)

    # Save to file
    save_path = RESULTS_DIR / "metrics" / "classification_report.txt"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        f.write("SMART RECYCLING CLASSIFIER — CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    print(f"💾 Report saved → {save_path}")

    return report


# ============================================================
# 📊 PER-CLASS ACCURACY BAR CHART
# ============================================================
def plot_per_class_accuracy(y_true, y_pred, phase="test"):
    """
    Bar chart of per-class accuracy

    Args:
        y_true : true labels
        y_pred : predicted labels
        phase  : label for filename
    """
    cm = confusion_matrix(y_true, y_pred)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    colors = ['#2ecc71' if acc >= 0.80 else
              '#f39c12' if acc >= 0.60 else
              '#e74c3c' for acc in per_class_acc]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(CLASS_NAMES, per_class_acc * 100, color=colors, edgecolor='white', linewidth=1.2)

    # Value labels on bars
    for bar, acc in zip(bars, per_class_acc):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f'{acc*100:.1f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    ax.set_title('Per-Class Accuracy', fontsize=15, fontweight='bold')
    ax.set_xlabel('Waste Category', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim(0, 115)
    ax.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='80% threshold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Color legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='≥ 80% (Good)'),
        Patch(facecolor='#f39c12', label='60–79% (Fair)'),
        Patch(facecolor='#e74c3c', label='< 60% (Needs Work)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()

    save_path = RESULTS_DIR / "plots" / f"per_class_accuracy_{phase}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Per-class accuracy chart saved → {save_path}")
    plt.close()

    return dict(zip(CLASS_NAMES, per_class_acc))


# ============================================================
# 🏆 OVERALL METRICS SUMMARY
# ============================================================
def print_metrics_summary(y_true, y_pred, y_proba):
    """
    Print overall evaluation metrics

    Args:
        y_true  : true labels
        y_pred  : predicted labels
        y_proba : softmax probabilities
    """
    overall_acc = accuracy_score(y_true, y_pred)
    top2_acc    = top_k_accuracy_score(y_true, y_proba, k=2)
    top3_acc    = top_k_accuracy_score(y_true, y_proba, k=3)

    print("\n" + "=" * 60)
    print("🏆 OVERALL METRICS SUMMARY")
    print("=" * 60)
    print(f"   Top-1 Accuracy  : {overall_acc*100:.2f}%")
    print(f"   Top-2 Accuracy  : {top2_acc*100:.2f}%")
    print(f"   Top-3 Accuracy  : {top3_acc*100:.2f}%")
    print("=" * 60)

    # Save metrics
    save_path = RESULTS_DIR / "metrics" / "overall_metrics.txt"
    with open(save_path, 'w') as f:
        f.write("SMART RECYCLING CLASSIFIER — OVERALL METRICS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Top-1 Accuracy  : {overall_acc*100:.2f}%\n")
        f.write(f"Top-2 Accuracy  : {top2_acc*100:.2f}%\n")
        f.write(f"Top-3 Accuracy  : {top3_acc*100:.2f}%\n")
    print(f"💾 Metrics saved → {save_path}")

    return {
        'top1_accuracy': overall_acc,
        'top2_accuracy': top2_acc,
        'top3_accuracy': top3_acc
    }


# ============================================================
# 🚀 FULL EVALUATION PIPELINE
# ============================================================
def run_evaluation(model_filename="recycle_net_final.keras"):
    """
    Full end-to-end evaluation pipeline

    Args:
        model_filename : model file inside MODELS_DIR
    """
    print("\n" + "=" * 60)
    print("📊 SMART RECYCLING CLASSIFIER — EVALUATION PIPELINE")
    print("=" * 60)

    # ── Step 1: Load model ────────────────────────────────────
    model = load_model(model_filename)

    # ── Step 2: Load test data ────────────────────────────────
    print("\n📂 Loading test generator...")
    _, _, test_gen = create_data_generators()
    print(f"   Test samples : {test_gen.samples}")

    # ── Step 3: Predictions ───────────────────────────────────
    y_true, y_pred, y_proba = get_predictions(model, test_gen)

    # ── Step 4: Overall metrics ───────────────────────────────
    metrics = print_metrics_summary(y_true, y_pred, y_proba)

    # ── Step 5: Classification report ────────────────────────
    print_classification_report(y_true, y_pred)

    # ── Step 6: Confusion matrix ──────────────────────────────
    plot_confusion_matrix(y_true, y_pred, normalize=True, phase="test")

    # ── Step 7: Per-class accuracy ────────────────────────────
    per_class = plot_per_class_accuracy(y_true, y_pred, phase="test")

    print("\n" + "=" * 60)
    print("🎉 EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"   Top-1 Accuracy : {metrics['top1_accuracy']*100:.2f}%")
    print(f"   Top-2 Accuracy : {metrics['top2_accuracy']*100:.2f}%")
    print("\n📁 Outputs saved to:")
    print(f"   {RESULTS_DIR / 'plots'}")
    print(f"   {RESULTS_DIR / 'metrics'}")
    print("=" * 60)

    return metrics, per_class


# ============================================================
# ▶️ ENTRY POINT
# ============================================================
if __name__ == "__main__":
    metrics, per_class = run_evaluation()
