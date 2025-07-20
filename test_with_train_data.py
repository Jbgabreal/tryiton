#!/usr/bin/env python3
"""
Test HR-VITON with available train data
"""

import os
import subprocess
import sys

def create_train_pairs():
    """Create a small test file using train data"""
    
    # Check if train data exists
    train_cloth_dir = "data/train/cloth"
    train_image_dir = "data/train/image"
    
    if not os.path.exists(train_cloth_dir):
        print("❌ Train cloth directory not found")
        return False
    
    if not os.path.exists(train_image_dir):
        print("❌ Train image directory not found")
        return False
    
    # Get some sample files
    cloth_files = [f for f in os.listdir(train_cloth_dir) if f.endswith('.jpg')][:5]
    image_files = [f for f in os.listdir(train_image_dir) if f.endswith('.jpg')][:5]
    
    if not cloth_files or not image_files:
        print("❌ No image files found in train directories")
        return False
    
    # Create test pairs file
    with open("data/train_pairs_small.txt", "w") as f:
        for i in range(min(len(cloth_files), len(image_files))):
            f.write(f"{image_files[i]} {cloth_files[i]}\n")
    
    print(f"✅ Created test file with {min(len(cloth_files), len(image_files))} pairs")
    return True

def run_train_test():
    """Run HR-VITON test with train data"""
    
    print("HR-VITON Test with Train Data")
    print("=" * 40)
    
    # Check if required files exist
    required_files = [
        'checkpoints/condition_generator.pth',
        'checkpoints/condition_discriminator.pth',
        'checkpoints/image_generator.pth',
        'eval_models/weights/v0.1/alex.pth'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Missing: {file_path}")
            return False
        else:
            print(f"✅ Found: {file_path}")
    
    # Create test pairs from train data
    if not create_train_pairs():
        return False
    
    print("\nStarting inference with train data...")
    
    # Run the test with train data
    cmd = [
        "python", "test_generator.py",
        "--occlusion",
        "--cuda", "False",
        "--test_name", "train_test",
        "--tocg_checkpoint", "checkpoints/condition_generator.pth",
        "--gpu_ids", "0",
        "--gen_checkpoint", "checkpoints/image_generator.pth",
        "--datasetting", "unpaired",
        "--dataroot", "./data",
        "--data_list", "train_pairs_small.txt"
    ]
    
    try:
        print("Running command:")
        print(" ".join(cmd))
        print("\nThis may take a few minutes...")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n🎉 Test completed successfully!")
            print("Results should be saved in: ./output/train_test/")
            return True
        else:
            print("\n❌ Test failed with error:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"\n❌ Error running test: {e}")
        return False

def main():
    success = run_train_test()
    
    if success:
        print("\n✅ Train data test completed!")
        print("You can now check the results in the output directory.")
    else:
        print("\n❌ Test failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 