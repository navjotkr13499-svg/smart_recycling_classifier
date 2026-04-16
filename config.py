"""
Central configuration file for Smart Recycling Classifier
"""
from pathlib import Path

# ============================================================
# 📁 PROJECT PATHS
# ============================================================
PROJECT_ROOT    = Path(__file__).parent
DATA_DIR        = PROJECT_ROOT / "data"
RAW_DATA_DIR    = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR     = PROJECT_ROOT / "results"
MODELS_DIR      = PROJECT_ROOT / "models"

# ============================================================
# 🏷️ DATASET SETTINGS
# ============================================================
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASS_NAMES)

# ============================================================
# 🖼️ IMAGE SETTINGS
# ============================================================
IMG_SIZE    = (224, 224)   # width, height
IMG_CHANNELS = 3            # RGB
INPUT_SHAPE = (*IMG_SIZE, IMG_CHANNELS)  # (224, 224, 3)

# ============================================================
# ✂️ DATA SPLIT RATIOS
# ============================================================
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# ============================================================
# 🏋️ TRAINING SETTINGS
# ============================================================
BATCH_SIZE   = 32
EPOCHS       = 50
RANDOM_SEED  = 42
LEARNING_RATE = 0.001

# ============================================================
# 🪄 AUGMENTATION SETTINGS
# ============================================================
AUGMENTATION_CONFIG = {
    'rotation_range'     : 20,
    'width_shift_range'  : 0.2,
    'height_shift_range' : 0.2,
    'shear_range'        : 0.2,
    'zoom_range'         : 0.2,
    'horizontal_flip'    : True,
    'fill_mode'          : 'nearest'
}

# ============================================================
# 🧠 MODEL SETTINGS
# ============================================================
MODEL_NAME       = "MobileNetV2"
DROPOUT_RATE     = 0.5
DENSE_UNITS      = 128
FINE_TUNE_LAYERS = 20          # number of layers to unfreeze for fine-tuning

# ============================================================
# 📁 CREATE DIRECTORIES (auto on import)
# ============================================================
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
                  RESULTS_DIR, MODELS_DIR,
                  RESULTS_DIR / "plots",
                  RESULTS_DIR / "metrics"]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================
# ✅ CONFIRM LOADED
# ============================================================
if __name__ == "__main__":
    print("✅ Config loaded successfully!")
    print(f"   Project Root : {PROJECT_ROOT}")
    print(f"   Data Dir     : {DATA_DIR}")
    print(f"   Image Size   : {IMG_SIZE}")
    print(f"   Classes      : {CLASS_NAMES}")
    print(f"   Batch Size   : {BATCH_SIZE}")
    print(f"   Epochs       : {EPOCHS}")
