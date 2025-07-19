#!/usr/bin/env python3
"""
Quick Start Script for HR-VITON
"""

import os
import sys
import argparse
import subprocess

def check_installation():
    """Check if HR-VITON is properly installed"""
    print("Checking HR-VITON installation...")
    
    # Check if required files exist
    required_files = [
        'checkpoints/condition_generator.pth',
        'checkpoints/image_generator.pth',
        'eval_models/weights/v0.1/alex.pth'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease download the model checkpoints first:")
        print("1. Run: python download_checkpoints.py")
        print("2. Or manually download from the links in setup_guide.md")
        return False
    
    print("✅ All required files found!")
    return True

def run_inference(person_image, clothing_image, output_dir="results"):
    """Run HR-VITON inference"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare command
    cmd = [
        "python", "test_generator.py",
        "--occlusion",
        "--cuda", "True",
        "--test_name", "quick_start",
        "--tocg_checkpoint", "checkpoints/condition_generator.pth",
        "--gpu_ids", "0",
        "--gen_checkpoint", "checkpoints/image_generator.pth",
        "--datasetting", "unpaired",
        "--dataroot", "./data",
        "--data_list", "quick_test_pairs.txt"
    ]
    
    # Create a simple test pairs file
    with open("quick_test_pairs.txt", "w") as f:
        f.write(f"{person_image} {clothing_image}\n")
    
    print(f"Running inference...")
    print(f"Person image: {person_image}")
    print(f"Clothing image: {clothing_image}")
    print(f"Output directory: {output_dir}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Inference completed successfully!")
            print(f"Results saved in: {output_dir}")
        else:
            print("❌ Inference failed:")
            print(result.stderr)
    except Exception as e:
        print(f"❌ Error running inference: {e}")

def main():
    parser = argparse.ArgumentParser(description="HR-VITON Quick Start")
    parser.add_argument("--person", type=str, help="Path to person image")
    parser.add_argument("--clothing", type=str, help="Path to clothing image")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    print("HR-VITON Quick Start")
    print("=" * 30)
    
    # Check installation
    if not check_installation():
        return
    
    # If no arguments provided, show usage
    if not args.person or not args.clothing:
        print("\nUsage:")
        print("python quick_start.py --person path/to/person.jpg --clothing path/to/clothing.jpg")
        print("\nExample:")
        print("python quick_start.py --person data/image/person1.jpg --clothing data/cloth/shirt1.jpg")
        print("\nNote: Make sure your images are in the correct format and directory structure.")
        return
    
    # Check if input files exist
    if not os.path.exists(args.person):
        print(f"❌ Person image not found: {args.person}")
        return
    
    if not os.path.exists(args.clothing):
        print(f"❌ Clothing image not found: {args.clothing}")
        return
    
    # Run inference
    run_inference(args.person, args.clothing, args.output)

if __name__ == "__main__":
    main() 