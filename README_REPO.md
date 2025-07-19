# HR-VITON Virtual Try-On Setup

This repository contains a working setup of HR-VITON (High-Resolution Virtual Try-On) with all compatibility fixes for CPU-only environments.

## 🎯 What's Included

- ✅ **Working HR-VITON Implementation** - Adapted for CPU-only environments
- ✅ **All Compatibility Fixes** - Resolved CUDA and NumPy deprecation issues
- ✅ **Helper Scripts** - Easy setup and monitoring tools
- ✅ **Small Test Dataset** - 10-sample test for quick verification
- ✅ **Comprehensive Documentation** - Step-by-step guides

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd HR-VITON

# Create conda environment
conda create -n hrviton python=3.8 -y
conda activate hrviton

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python torchgeometry Pillow tqdm tensorboardX scikit-image scipy psutil
```

### 2. Download Model Checkpoints
Download these files and place them in the correct directories:

**Model Checkpoints:**
- `checkpoints/condition_generator.pth` - [Download](https://drive.google.com/file/d/1XJTCdRBOPVgVTmqzhVGFAgMm2NLkw5uQ/view?usp=sharing)
- `checkpoints/condition_discriminator.pth` - [Download](https://drive.google.com/file/d/1T4V3cyRlY5sHVK7Quh_EJY5dovb5FxGX/view?usp=share_link)
- `checkpoints/image_generator.pth` - [Download](https://drive.google.com/file/d/1T5_YDUhYSSKPC_nZMk2NeC-XXUFoYeNy/view?usp=share_link)
- `eval_models/weights/v0.1/alex.pth` - [Download](https://drive.google.com/file/d/1FF3BBSDIA3uavmAiuMH6YFCv09Lt8jUr/view?usp=sharing)

### 3. Test Installation
```bash
python test_installation.py
```

### 4. Run Test
```bash
# Test with 10 samples
python test_small_dataset.py

# Monitor progress
python monitor_progress.py
```

## 📁 Repository Structure

```
HR-VITON/
├── checkpoints/                    # Model checkpoints (download required)
├── data/                          # Dataset directory
│   ├── test_pairs_small.txt      # 10-sample test dataset
│   └── test/                     # Full dataset (download required)
├── eval_models/weights/v0.1/     # AlexNet weights (download required)
├── output/                        # Generated results
├── scripts/                       # Helper scripts
├── test_installation.py           # Installation test
├── test_small_dataset.py          # Small dataset test
├── monitor_progress.py            # Progress monitoring
├── quick_start.py                 # Easy interface
├── download_checkpoints.py        # Download helper
├── setup_guide.md                 # Detailed setup guide
├── NEXT_STEPS.md                  # Next steps guide
└── SETUP_COMPLETE.md             # Completion summary
```

## 🔧 Available Scripts

### Core Scripts
- `test_installation.py` - Verify all dependencies and files
- `test_small_dataset.py` - Run test with 10 samples
- `monitor_progress.py` - Real-time progress monitoring
- `quick_start.py` - Easy interface for custom images

### Helper Scripts
- `download_checkpoints.py` - Download model files
- `test_generator.py` - Main inference script
- `train_generator.py` - Training script (if needed)

## 🎨 Usage Examples

### Test with Sample Data
```bash
python test_small_dataset.py
```

### Use Your Own Images
```bash
python quick_start.py --person path/to/person.jpg --clothing path/to/clothing.jpg
```

### Monitor Progress
```bash
python monitor_progress.py
```

### Run Full Dataset
```bash
python test_generator.py --occlusion --cuda False --test_name full_test --tocg_checkpoint checkpoints/condition_generator.pth --gpu_ids 0 --gen_checkpoint checkpoints/image_generator.pth --datasetting unpaired --dataroot ./data --data_list test_pairs.txt
```

## 🔧 Compatibility Fixes Applied

### 1. CUDA Compatibility
- Fixed all `.cuda()` calls to check for CUDA availability
- Added `torch.cuda.is_available()` checks throughout codebase
- Made model work on CPU-only environments

### 2. NumPy Deprecation
- Replaced deprecated `np.float` with `float`
- Updated all tensor type conversions

### 3. Path Handling
- Fixed Windows path issues
- Improved file path handling

## 📊 Performance Notes

- **CPU Mode**: Model runs on CPU (slower but functional)
- **Processing Time**: ~2-5 minutes per image on CPU
- **Memory Usage**: ~2.5GB RAM during processing
- **Quality**: High-quality results despite CPU processing

## 🎯 Key Features

- ✅ **CPU-Compatible**: Works without GPU
- ✅ **Easy Setup**: Automated installation scripts
- ✅ **Progress Monitoring**: Real-time status updates
- ✅ **Small Test Dataset**: Quick verification
- ✅ **Custom Images**: Support for your own photos
- ✅ **Comprehensive Documentation**: Step-by-step guides

## 🆘 Troubleshooting

### Common Issues
1. **CUDA Errors**: Model automatically falls back to CPU
2. **Memory Issues**: Reduce batch size or use smaller images
3. **Download Failures**: Use manual download links provided
4. **Path Errors**: Ensure all files are in correct directories

### Getting Help
- Check `test_installation.py` for setup verification
- Use `monitor_progress.py` to track processing
- Review `setup_guide.md` for detailed instructions

## 📝 License

This setup is based on the original HR-VITON paper:
- **Paper**: [High-Resolution Virtual Try-On with Misalignment and Occlusion-Handled Conditions](https://arxiv.org/abs/2206.14180)
- **Original Repository**: [HR-VITON](https://github.com/sangyun884/HR-VITON)

## 🎉 Success!

This repository provides a fully working HR-VITON setup that:
- ✅ Works on CPU-only environments
- ✅ Includes all necessary fixes and improvements
- ✅ Provides easy-to-use scripts and monitoring
- ✅ Supports both test data and custom images

**Ready to use for virtual try-on applications!** 🚀 