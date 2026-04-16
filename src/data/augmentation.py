"""
Data augmentation utilities
"""
import os
import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# ✅ Fix: explicitly point to project root BEFORE any imports
PROJECT_ROOT = Path(__file__).parent.parent.parent  # goes up: data → src → project root
sys.path.insert(0, str(PROJECT_ROOT))               # insert at position 0 (highest priority)

# ✅ Now safely import your config
from config import *


def create_data_generators(batch_size=BATCH_SIZE):
    """
    Create data generators for train, validation, and test sets
    
    Returns:
        train_gen, val_gen, test_gen: Data generators
    """
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=AUGMENTATION_CONFIG['rotation_range'],
        width_shift_range=AUGMENTATION_CONFIG['width_shift_range'],
        height_shift_range=AUGMENTATION_CONFIG['height_shift_range'],
        shear_range=AUGMENTATION_CONFIG['shear_range'],
        zoom_range=AUGMENTATION_CONFIG['zoom_range'],
        horizontal_flip=AUGMENTATION_CONFIG['horizontal_flip'],
        fill_mode=AUGMENTATION_CONFIG['fill_mode']
    )
    
    # Validation and test data generators (only rescaling)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        PROCESSED_DATA_DIR / 'train',
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=RANDOM_SEED
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        PROCESSED_DATA_DIR / 'val',
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        PROCESSED_DATA_DIR / 'test',
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator, test_generator

def visualize_augmentation(generator, save_path=None):
    """
    Visualize augmented images
    
    Args:
        generator: Image data generator
        save_path: Path to save the visualization
    """
    import matplotlib.pyplot as plt
    
    # Get a batch of images
    images, labels = next(generator)
    
    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Augmented Training Images', fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            class_idx = labels[i].argmax()
            ax.set_title(CLASS_NAMES[class_idx])
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved to {save_path}")
    
    plt.show()

if __name__ == "__main__":
    print("🔄 Creating data generators...")
    
    train_gen, val_gen, test_gen = create_data_generators()
    
    print(f"\n📊 Data Generator Summary:")
    print(f"  Train samples: {train_gen.samples}")
    print(f"  Val samples: {val_gen.samples}")
    print(f"  Test samples: {test_gen.samples}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Classes: {train_gen.class_indices}")
    
    # Visualize
    print("\n🎨 Visualizing augmented images...")
    visualize_augmentation(train_gen, RESULTS_DIR / 'plots' / 'augmented_samples.png')