#!/usr/bin/env python3
"""
Test HR-VITON with small dataset (10 samples)
"""

import os
import subprocess
import sys

def run_small_test():
    """Run HR-VITON test with small dataset"""
    
    print("HR-VITON Small Dataset Test (10 samples)")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [
        'checkpoints/condition_generator.pth',
        'checkpoints/condition_discriminator.pth',
        'checkpoints/image_generator.pth',
        'eval_models/weights/v0.1/alex.pth',
        'data/test_pairs_small.txt'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Missing: {file_path}")
            return False
        else:
            print(f"✅ Found: {file_path}")
    
    print("\nStarting inference with small dataset...")
    
    # Run the test with small dataset
    cmd = [
        "python", "test_generator.py",
        "--occlusion",
        "--cuda", "False",
        "--test_name", "small_test",
        "--tocg_checkpoint", "checkpoints/condition_generator.pth",
        "--gpu_ids", "0",
        "--gen_checkpoint", "checkpoints/image_generator.pth",
        "--datasetting", "unpaired",
        "--dataroot", "./data",
        "--data_list", "test_pairs_small.txt"
    ]
    
    try:
        print("Running command:")
        print(" ".join(cmd))
        print("\nThis may take a few minutes...")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n🎉 Test completed successfully!")
            print("Results should be saved in: ./results/small_test/")
            return True
        else:
            print("\n❌ Test failed with error:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"\n❌ Error running test: {e}")
        return False

def main():
    success = run_small_test()
    
    if success:
        print("\n✅ Small dataset test completed!")
        print("You can now run the full dataset or use your own images.")
    else:
        print("\n❌ Test failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 