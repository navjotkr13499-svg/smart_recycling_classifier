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

    url = "https://github.com/garythung/trashnet/archive/refs/heads/master.zip"

    data_dir = Path("data/raw")
    zip_path = data_dir / "trashnet.zip"
    extract_path = data_dir / "trashnet_extracted"

    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ✅ Step 1: Download master zip
        urllib.request.urlretrieve(url, zip_path)
        print("✅ Download complete!")

        # ✅ Step 2: Extract master zip
        print("📦 Extracting master zip...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("✅ Extraction complete!")

        # ✅ Step 3: Debug — show what's actually inside
        print("\n🔎 Scanning extracted contents...")
        for item in extract_path.rglob("*"):
            print(f"   {item}")

        # ✅ Step 4: Find and extract nested dataset-resized.zip
        nested_zip = None
        for item in extract_path.rglob("*.zip"):
            nested_zip = item
            print(f"\n📦 Found nested zip: {nested_zip}")
            break

        if nested_zip:
            nested_extract = extract_path / "dataset_resized"
            with zipfile.ZipFile(nested_zip, 'r') as zip_ref:
                zip_ref.extractall(nested_extract)
            print("✅ Nested zip extracted!")

            # ✅ Step 5: Find category folders
            print("\n📁 Organizing dataset...")
            for folder in nested_extract.rglob("*"):
                if folder.is_dir() and folder.name in ["cardboard", "glass", "metal", "paper", "plastic", "trash"]:
                    dst = data_dir / folder.name
                    dst.mkdir(parents=True, exist_ok=True)
                    images_copied = 0
                    for img in folder.iterdir():
                        if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            shutil.copy(img, dst / img.name)
                            images_copied += 1
                    print(f"   ✅ {folder.name}: {images_copied} images copied")
        else:
            # ✅ Fallback: Try direct folder copy
            print("\n📁 No nested zip found, trying direct folder copy...")
            source_dir = extract_path / "trashnet-master" / "data"
            if source_dir.exists():
                for category in os.listdir(source_dir):
                    src = source_dir / category
                    dst = data_dir / category
                    if src.is_dir():
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                        print(f"   ✅ {category}: {len(os.listdir(dst))} images")
            else:
                print(f"❌ Source dir not found: {source_dir}")

        # ✅ Step 6: Cleanup
        print("\n🧹 Cleaning up temp files...")
        os.remove(zip_path)
        shutil.rmtree(extract_path)
        print("✅ Cleanup done!")

        print("\n✨ Dataset ready in data/raw/")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n📌 Manual download instructions:")
        print("1. Go to: https://github.com/garythung/trashnet")
        print("2. Download dataset-resized.zip from the repo")
        print("3. Extract category folders into data/raw/")

def verify_dataset():
    """
    Verify dataset structure and count images
    """
    print("\n🔍 Verifying dataset...")

    data_dir = Path("data/raw")
    categories = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

    total_images = 0
    all_found = True

    for category in categories:
        cat_path = data_dir / category
        if cat_path.exists():
            count = len([
                f for f in cat_path.iterdir()
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ])
            print(f"   ✅ {category.capitalize():<12}: {count} images")
            total_images += count
        else:
            print(f"   ⚠️  {category.capitalize():<12}: folder not found")
            all_found = False

    print(f"\n📊 Total images: {total_images}")

    if all_found and total_images > 0:
        print("🎉 Dataset is complete and ready!")
    else:
        print("⚠️  Some categories are missing. Check the output above.")

    return total_images > 0

if __name__ == "__main__":
    download_trashnet()
    verify_dataset()
