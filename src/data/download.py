"""
Script to download and organize the TrashNet dataset
"""
import os
import urllib.request
import zipfile
from pathlib import Path
import shutil

def download_trashnet():
    """
    Download TrashNet dataset from GitHub
    """
    print("📥 Downloading TrashNet dataset...")
    
    # URLs for dataset
    url = "https://github.com/garythung/trashnet/archive/refs/heads/master.zip"
    
    # Paths
    data_dir = Path("data/raw")
    zip_path = data_dir / "trashnet.zip"
    extract_path = data_dir / "trashnet_extracted"
    
    # Create directories
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("✅ Download complete!")
        
        # Extract
        print("📦 Extracting files...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        print("✅ Extraction complete!")
        
        # Organize files
        print("📁 Organizing dataset...")
        source_dir = extract_path / "trashnet-master" / "data"
        
        if source_dir.exists():
            # Copy to organized structure
            for category in os.listdir(source_dir):
                src = source_dir / category
                dst = data_dir / category
                if src.is_dir():
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                    print(f"   Copied {category}: {len(os.listdir(dst))} images")
        
        # Cleanup
        print("🧹 Cleaning up...")
        os.remove(zip_path)
        shutil.rmtree(extract_path)
        
        print("\n✨ Dataset ready in data/raw/")
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n📌 Manual download instructions:")
        print("1. Go to: https://github.com/garythung/trashnet")
        print("2. Download the repository")
        print("3. Extract to data/raw/")

def verify_dataset():
    """
    Verify dataset structure and count images
    """
    print("\n🔍 Verifying dataset...")
    
    data_dir = Path("data/raw")
    categories = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    
    total_images = 0
    for category in categories:
        cat_path = data_dir / category
        if cat_path.exists():
            count = len([f for f in cat_path.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
            print(f"   {category.capitalize()}: {count} images")
            total_images += count
        else:
            print(f"   ⚠️  {category.capitalize()}: folder not found")
    
    print(f"\n📊 Total images: {total_images}")
    return total_images > 0

if __name__ == "__main__":
    download_trashnet()
    verify_dataset()