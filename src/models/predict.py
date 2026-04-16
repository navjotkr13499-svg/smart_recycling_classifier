"""
Prediction pipeline for Smart Recycling Classifier
Includes: single image, batch prediction, confidence visualization
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from PIL import Image
from config import *
from src.models.model import load_model


# ============================================================
# 🖼️ IMAGE PREPROCESSING
# ============================================================
def preprocess_image(image_path, target_size=IMG_SIZE):
    """
    Load and preprocess a single image for inference

    Args:
        image_path  : str or Path to image file
        target_size : tuple (H, W) — default (224, 224)

    Returns:
        img_array   : preprocessed numpy array (1, H, W, 3)
        img_display : original PIL image for visualization
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"❌ Image not found: {image_path}")

    # Load & resize
    img_display = Image.open(image_path).convert('RGB')
    img_resized = img_display.resize(target_size)

    # Normalize to [0, 1]
    img_array = np.array(img_resized) / 255.0

    # Add batch dimension → (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0).astype('float32')

    return img_array


# ============================================================
# 🎯 SINGLE IMAGE PREDICTION
# ============================================================
def predict_single(model, image_path, top_k=3):
    """
    Predict waste category for a single image

    Args:
        model      : trained Keras model
        image_path : path to image
        top_k      : number of top predictions to return

    Returns:
        result : dict with prediction details
    """
    # Preprocess
    img_array, img_display = preprocess_image(image_path)

    # Inference
    proba = model.predict(img_array, verbose=0)[0]  # shape (6,)

    # Top-K predictions
    top_indices = np.argsort(proba)[::-1][:top_k]
    top_classes = [CLASS_NAMES[i] for i in top_indices]
    top_probs   = [float(proba[i]) for i in top_indices]

    result = {
        'image_path'     : str(image_path),
        'predicted_class': top_classes[0],
        'confidence'     : top_probs[0],
        'top_k'          : list(zip(top_classes, top_probs)),
        'all_probs'      : dict(zip(CLASS_NAMES, proba.tolist())),
        'img_display'    : img_display
    }

    return result


# ============================================================
# 📊 VISUALIZE SINGLE PREDICTION
# ============================================================
def visualize_prediction(result, save_path=None, show=False):
    """
    Visualize image + confidence bar chart side by side

    Args:
        result    : dict from predict_single()
        save_path : optional path to save figure
        show      : whether to call plt.show()
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('♻️ Smart Recycling Classifier — Prediction',
                 fontsize=15, fontweight='bold')

    # ── Left: Image ───────────────────────────────────────────
    axes[0].imshow(result['img_display'])
    axes[0].axis('off')

    conf  = result['confidence']
    label = result['predicted_class'].upper()
    color = '#2ecc71' if conf >= 0.80 else '#f39c12' if conf >= 0.50 else '#e74c3c'

    axes[0].set_title(
        f"Predicted: {label}\nConfidence: {conf*100:.1f}%",
        fontsize=13, fontweight='bold', color=color
    )

    # ── Right: All class probabilities ────────────────────────
    classes = CLASS_NAMES
    probs   = [result['all_probs'][c] * 100 for c in classes]
    colors  = [
        '#2ecc71' if c == result['predicted_class'] else '#aed6f1'
        for c in classes
    ]

    bars = axes[1].barh(classes, probs, color=colors,
                        edgecolor='white', linewidth=1.0)

    # Value labels
    for bar, prob in zip(bars, probs):
        axes[1].text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f'{prob:.1f}%',
            va='center', ha='left', fontsize=10
        )

    axes[1].set_title('Confidence per Class', fontsize=13)
    axes[1].set_xlabel('Confidence (%)', fontsize=11)
    axes[1].set_xlim(0, 115)
    axes[1].grid(axis='x', alpha=0.3)

    # Legend
    legend = [
        mpatches.Patch(color='#2ecc71', label='Predicted Class'),
        mpatches.Patch(color='#aed6f1', label='Other Classes')
    ]
    axes[1].legend(handles=legend, loc='lower right')

    plt.tight_layout()

    # Save
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Prediction plot saved → {save_path}")

    if show:
        plt.show()

    plt.close()


# ============================================================
# 📁 BATCH PREDICTION
# ============================================================
def predict_batch(model, image_dir, extensions=('.jpg', '.jpeg', '.png')):
    """
    Run predictions on all images in a directory

    Args:
        model      : trained Keras model
        image_dir  : path to directory of images
        extensions : image file types to include

    Returns:
        results : list of prediction dicts
    """
    image_dir = Path(image_dir)
    images    = [p for p in image_dir.rglob('*') if p.suffix.lower() in extensions]

    if not images:
        print(f"⚠️  No images found in {image_dir}")
        return []

    print(f"\n📁 Batch prediction on {len(images)} images...")
    results = []

    for i, img_path in enumerate(images, 1):
        try:
            result = predict_single(model, img_path)
            results.append(result)
            print(f"   [{i:>3}/{len(images)}] {img_path.name:<30} "
                  f"→ {result['predicted_class']:<12} "
                  f"({result['confidence']*100:.1f}%)")
        except Exception as e:
            print(f"   ❌ Failed on {img_path.name}: {e}")

    return results


# ============================================================
# 📊 BATCH SUMMARY CHART
# ============================================================
def plot_batch_summary(results, save_path=None):
    """
    Pie chart + confidence distribution for batch results

    Args:
        results   : list of dicts from predict_batch()
        save_path : optional path to save figure
    """
    from collections import Counter

    if not results:
        print("⚠️  No results to plot.")
        return

    # Count predictions
    pred_classes = [r['predicted_class'] for r in results]
    counts       = Counter(pred_classes)
    confidences  = [r['confidence'] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('♻️ Batch Prediction Summary', fontsize=15, fontweight='bold')

    # ── Pie chart ─────────────────────────────────────────────
    pie_colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
    axes[0].pie(
        counts.values(),
        labels=counts.keys(),
        autopct='%1.1f%%',
        colors=pie_colors,
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
    )
    axes[0].set_title(f'Class Distribution\n(n={len(results)} images)', fontsize=13)

    # ── Confidence histogram ──────────────────────────────────
    axes[1].hist(
        [c * 100 for c in confidences],
        bins=20,
        color='#3498db',
        edgecolor='white',
        linewidth=0.8
    )
    avg_conf = np.mean(confidences) * 100
    axes[1].axvline(avg_conf, color='red', linestyle='--',
                    label=f'Mean: {avg_conf:.1f}%')
    axes[1].set_title('Confidence Distribution', fontsize=13)
    axes[1].set_xlabel('Confidence (%)', fontsize=11)
    axes[1].set_ylabel('Number of Images', fontsize=11)
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Batch summary saved → {save_path}")

    plt.close()


# ============================================================
# ▶️ ENTRY POINT — Quick demo
# ============================================================
if __name__ == "__main__":
    import random

    print("=" * 60)
    print("🎯 SMART RECYCLING CLASSIFIER — PREDICTION DEMO")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────
    model = load_model("recycle_net_final.keras")

    # ── Pick a random test image ──────────────────────────────
    test_dir = DATA_DIR / "test"
    all_images = list(test_dir.rglob("*.jpg"))

    if not all_images:
        print("⚠️  No test images found. Run preprocess.py first.")
        sys.exit(1)

    sample_image = random.choice(all_images)
    true_label   = sample_image.parent.name

    print(f"\n🖼️  Sample image : {sample_image.name}")
    print(f"   True label   : {true_label}")

    # ── Single prediction ─────────────────────────────────────
    result = predict_single(model, sample_image, top_k=3)

    print(f"\n🎯 Prediction Results:")
    print(f"   Predicted    : {result['predicted_class']}")
    print(f"   Confidence   : {result['confidence']*100:.2f}%")
    print(f"   Correct      : {'✅' if result['predicted_class'] == true_label else '❌'}")
    print(f"\n   Top-3 Predictions:")
    for cls, prob in result['top_k']:
        bar = '█' * int(prob * 30)
        print(f"   {cls:<12} {bar:<30} {prob*100:.1f}%")

    # ── Visualize ─────────────────────────────────────────────
    save_path = RESULTS_DIR / "plots" / "sample_prediction.png"
    visualize_prediction(result, save_path=str(save_path))

    print(f"\n✅ Demo complete!")
    print(f"   Plot saved → {save_path}")
    print("=" * 60)
