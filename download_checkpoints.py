#!/usr/bin/env python3
"""
Script to download HR-VITON model checkpoints
"""

import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    """Download a file with progress bar"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filename, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✅ Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def main():
    print("HR-VITON Model Checkpoint Downloader")
    print("=" * 50)
    
    # Create directories if they don't exist
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("eval_models/weights/v0.1", exist_ok=True)
    
    # Model checkpoint URLs and filenames
    checkpoints = {
        "checkpoints/condition_generator.pth": "https://drive.google.com/uc?export=download&id=1XJTCdRBOPVgVTmqzhVGFAgMm2NLkw5uQ",
        "checkpoints/condition_discriminator.pth": "https://drive.google.com/uc?export=download&id=1T4V3cyRlY5sHVK7Quh_EJY5dovb5FxGX", 
        "checkpoints/image_generator.pth": "https://drive.google.com/uc?export=download&id=1T5_YDUhYSSKPC_nZMk2NeC-XXUFoYeNy",
        "eval_models/weights/v0.1/alex.pth": "https://drive.google.com/uc?export=download&id=1FF3BBSDIA3uavmAiuMH6YFCv09Lt8jUr"
    }
    
    print("Note: These are Google Drive links. You may need to:")
    print("1. Open the links in a browser")
    print("2. Download manually if automatic download fails")
    print("3. Place the files in the correct directories")
    print()
    
    for filename, url in checkpoints.items():
        if os.path.exists(filename):
            print(f"✅ {filename} already exists")
        else:
            print(f"📥 Downloading {filename}...")
            print(f"   URL: {url}")
            print(f"   Manual download link: https://drive.google.com/file/d/{url.split('id=')[1]}/view")
            print()
    
    print("=" * 50)
    print("Manual Download Instructions:")
    print("1. Visit each Google Drive link above")
    print("2. Download the files")
    print("3. Place them in the correct directories:")
    print("   - checkpoints/condition_generator.pth")
    print("   - checkpoints/condition_discriminator.pth") 
    print("   - checkpoints/image_generator.pth")
    print("   - eval_models/weights/v0.1/alex.pth")
    print()
    print("After downloading, run: python test_installation.py")

if __name__ == "__main__":
    main() 