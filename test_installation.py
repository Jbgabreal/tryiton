#!/usr/bin/env python3
"""
Test script to verify HR-VITON installation
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'torch',
        'torchvision', 
        'torchaudio',
        'cv2',  # opencv-python
        'torchgeometry',
        'PIL',  # Pillow
        'tqdm',
        'tensorboardX',
        'skimage',  # scikit-image
        'scipy'
    ]
    
    print("Testing imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {failed_imports}")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_cuda():
    """Test CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA not available - model will run on CPU")
            return False
    except Exception as e:
        print(f"❌ CUDA test failed: {e}")
        return False

def test_model_files():
    """Test if model files exist"""
    import os
    
    required_files = [
        'checkpoints/condition_generator.pth',
        'checkpoints/condition_discriminator.pth',
        'checkpoints/image_generator.pth', 
        'eval_models/weights/v0.1/alex.pth'
    ]
    
    print("\nChecking model files...")
    missing_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (not found)")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        print("   Please download the model checkpoints as described in setup_guide.md")
        return False
    else:
        print("\n✅ All model files found!")
        return True

def main():
    print("HR-VITON Installation Test")
    print("=" * 40)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test CUDA
    cuda_ok = test_cuda()
    
    # Test model files
    files_ok = test_model_files()
    
    print("\n" + "=" * 40)
    if imports_ok and cuda_ok and files_ok:
        print("🎉 All tests passed! HR-VITON is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return imports_ok and cuda_ok and files_ok

if __name__ == "__main__":
    main() 