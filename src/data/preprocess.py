"""
Data preprocessing and train/val/test split
"""
import os
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *

def create_directory_structure():
    """Create directories for processed data"""
    splits = ['train', 'val', 'test']
    
    for split in splits:
        for category in CATEGORIES:
            path = PROCESSED_DATA_DIR / split / category
            path.mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created")

def split_dataset(test_size=TEST_SPLIT, val_size=VALIDATION_SPLIT, random_state=RANDOM_SEED):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining after test split)
        random_state: Random seed for reproducibility
    """
    print("\n🔀 Splitting dataset...")
    
    # Create directory structure
    create_directory_structure()
    
    stats = {
        'train': {},
        'val': {},
        'test': {}
    }
    
    for category in CATEGORIES:
        print(f"\n📂 Processing {category}...")
        
        # Get all images for this category
        category_path = RAW_DATA_DIR / category
        if not category_path.exists():
            print(f"⚠️  {category} folder not found, skipping...")
            continue
        
        images = list(category_path.glob("*.jpg")) + list(category_path.glob("*.png"))
        
        if len(images) == 0:
            print(f"⚠️  No images found in {category}, skipping...")
            continue
        
        print(f"   Found {len(images)} images")
        
        # First split: separate test set
        train_val_images, test_images = train_test_split(
            images,
            test_size=test_size,
            random_state=random_state
        )
        
        # Second split: separate validation from training
        val_ratio = val_size / (1 - test_size)  # Adjust ratio
        train_images, val_images = train_test_split(
            train_val_images,
            test_size=val_ratio,
            random_state=random_state
        )
        
        # Copy files to respective directories
        splits_data = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }
        
        for split_name, image_list in splits_data.items():
            dest_dir = PROCESSED_DATA_DIR / split_name / category
            
            for img_path in tqdm(image_list, desc=f"   Copying to {split_name}", leave=False):
                dest_path = dest_dir / img_path.name
                shutil.copy2(img_path, dest_path)
            
            stats[split_name][category] = len(image_list)
            print(f"   {split_name}: {len(image_list)} images")
    
    return stats

def print_split_summary(stats):
    """Print summary of the data split"""
    print("\n" + "="*60)
    print("📊 DATA SPLIT SUMMARY")
    print("="*60)
    
    for split_name in ['train', 'val', 'test']:
        print(f"\n{split_name.upper()} SET:")
        total = sum(stats[split_name].values())
        for category, count in stats[split_name].items():
            percentage = (count / total * 100) if total > 0 else 0
            print(f"  {category:.<20} {count:>5} ({percentage:>5.1f}%)")
        print(f"  {'TOTAL':.<20} {total:>5}")
    
    print("\n" + "="*60)

def verify_split():
    """Verify that all files are correctly split"""
    print("\n🔍 Verifying split...")
    
    for split in ['train', 'val', 'test']:
        split_path = PROCESSED_DATA_DIR / split
        if not split_path.exists():
            print(f"❌ {split} directory not found!")
            continue
        
        total = 0
        for category in CATEGORIES:
            cat_path = split_path / category
            if cat_path.exists():
                count = len(list(cat_path.glob("*.jpg"))) + len(list(cat_path.glob("*.png")))
                total += count
        
        print(f"✅ {split}: {total} images")

if __name__ == "__main__":
    print("🚀 Starting data preprocessing...")
    
    # Check if raw data exists
    if not RAW_DATA_DIR.exists():
        print("❌ Raw data directory not found!")
        print("Please run 'python src/data/download.py' first")
        exit(1)
    
    # Split dataset
    stats = split_dataset()
    
    # Print summary
    print_split_summary(stats)
    
    # Verify
    verify_split()
    
    print("\n✨ Data preprocessing complete!")
    print(f"📁 Processed data saved to: {PROCESSED_DATA_DIR}")