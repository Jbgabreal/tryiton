# HR-VITON Setup - Next Steps

## ✅ What's Complete
- ✅ Repository cloned successfully
- ✅ Conda environment `hrviton` created with Python 3.8
- ✅ All required Python packages installed:
  - PyTorch, torchvision, torchaudio
  - OpenCV, torchgeometry, Pillow
  - tqdm, tensorboardX, scikit-image, scipy
- ✅ Directory structure created
- ✅ Test scripts created

## 🔄 What's In Progress
- ⏳ Model checkpoint downloads (manual process required)

## 📋 Next Steps to Complete Setup

### 1. Download Model Checkpoints
You need to download 4 model files. Run this command to see the download links:

```bash
python download_checkpoints.py
```

**Manual Download Links:**
- **Condition Generator**: https://drive.google.com/file/d/1XJTCdRBOPVgVTmqzhVGFAgMm2NLkw5uQ/view?usp=sharing
- **Condition Discriminator**: https://drive.google.com/file/d/1T4V3cyRlY5sHVK7Quh_EJY5dovb5FxGX/view?usp=share_link
- **Image Generator**: https://drive.google.com/file/d/1T5_YDUhYSSKPC_nZMk2NeC-XXUFoYeNy/view?usp=share_link
- **AlexNet (LPIPS)**: https://drive.google.com/file/d/1FF3BBSDIA3uavmAiuMH6YFCv09Lt8jUr/view?usp=sharing

**Save files as:**
- `checkpoints/condition_generator.pth`
- `checkpoints/condition_discriminator.pth`
- `checkpoints/image_generator.pth`
- `eval_models/weights/v0.1/alex.pth`

### 2. Download Dataset (Optional for Testing)
For testing with sample data, download the VITON-HD dataset:
https://github.com/shadow2496/VITON-HD

Place it in the `./data` directory.

### 3. Test Installation
After downloading checkpoints:

```bash
python test_installation.py
```

### 4. Run Your First Try-On
Once everything is set up:

```bash
python quick_start.py --person path/to/person.jpg --clothing path/to/clothing.jpg
```

## 🎯 Quick Commands

**Check installation:**
```bash
python test_installation.py
```

**See download links:**
```bash
python download_checkpoints.py
```

**Run inference:**
```bash
python quick_start.py --person data/image/person1.jpg --clothing data/cloth/shirt1.jpg
```

## 📁 Current Directory Structure
```
HR-VITON/
├── checkpoints/                    # Model checkpoints (to be downloaded)
├── data/                          # Dataset directory
├── eval_models/weights/v0.1/      # AlexNet weights (to be downloaded)
├── test_installation.py           # Installation test script
├── download_checkpoints.py        # Download helper script
├── quick_start.py                 # Quick start script
├── setup_guide.md                 # Detailed setup guide
└── NEXT_STEPS.md                  # This file
```

## ⚠️ Important Notes
- The model will run on CPU if CUDA is not available (which is fine for testing)
- Make sure you have enough disk space for the model checkpoints (~2GB total)
- For best performance, use a GPU with CUDA support

## 🆘 Troubleshooting
- If downloads fail, try opening the Google Drive links in a browser
- If CUDA errors occur, the model will automatically fall back to CPU
- Check the `setup_guide.md` for detailed troubleshooting steps

## 🚀 Ready to Go!
Once you've downloaded the model checkpoints, you'll be ready to run virtual try-on with HR-VITON! 