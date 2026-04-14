"""
Configuration file for the project
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Dataset
CATEGORIES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CATEGORIES)

# Class mapping
CLASS_NAMES = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash"
}

# Image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Training parameters
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# Data augmentation parameters
AUGMENTATION_CONFIG = {
    'rotation_range': 20,
    'width_shift_range': 0.2,
    'height_shift_range': 0.2,
    'shear_range': 0.2,
    'zoom_range': 0.2,
    'horizontal_flip': True,
    'fill_mode': 'nearest'
}

# Model parameters
MODEL_NAME = 'efficientnet_b0'
PRETRAINED_WEIGHTS = 'imagenet'
DROPOUT_RATE = 0.5

# Recycling information
RECYCLING_INFO = {
    "cardboard": {
        "recyclable": True,
        "instructions": "Flatten boxes. Remove tape and labels. Keep dry.",
        "tips": "Pizza boxes with grease are NOT recyclable - they're compost.",
        "bin_color": "blue"
    },
    "glass": {
        "recyclable": True,
        "instructions": "Rinse container. Remove lids. Check local rules for colors.",
        "tips": "Broken glass may need special handling - check local guidelines.",
        "bin_color": "blue"
    },
    "metal": {
        "recyclable": True,
        "instructions": "Rinse cans. Remove labels if possible. Crush to save space.",
        "tips": "Aluminum cans are infinitely recyclable!",
        "bin_color": "blue"
    },
    "paper": {
        "recyclable": True,
        "instructions": "Keep dry and clean. Shredded paper in paper bag.",
        "tips": "Wax-coated paper (like some cups) is NOT recyclable.",
        "bin_color": "blue"
    },
    "plastic": {
        "recyclable": "depends",
        "instructions": "Check recycling number (1,2,5 usually accepted). Rinse clean.",
        "tips": "Look for ♻️ symbol with number 1-7. Not all plastics are recyclable!",
        "bin_color": "blue"
    },
    "trash": {
        "recyclable": False,
        "instructions": "Goes to landfill. Consider if item can be reused or donated first.",
        "tips": "When in doubt, don't contaminate recycling - put in trash.",
        "bin_color": "black"
    }
}