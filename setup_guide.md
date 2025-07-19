# HR-VITON Setup Guide

## Environment Setup ✅
- Conda environment `hrviton` created with Python 3.8
- PyTorch installation in progress
- Additional dependencies installation in progress

## Next Steps Required:

### 1. Download Model Checkpoints
You need to download the following model checkpoints:

**Try-on condition generator:**
- Link: https://drive.google.com/file/d/1XJTCdRBOPVgVTmqzhVGFAgMm2NLkw5uQ/view?usp=sharing
- Save as: `checkpoints/condition_generator.pth`

**Try-on condition generator (discriminator):**
- Link: https://drive.google.com/file/d/1T4V3cyRlY5sHVK7Quh_EJY5dovb5FxGX/view?usp=share_link
- Save as: `checkpoints/condition_discriminator.pth`

**Try-on image generator:**
- Link: https://drive.google.com/file/d/1T5_YDUhYSSKPC_nZMk2NeC-XXUFoYeNy/view?usp=share_link
- Save as: `checkpoints/image_generator.pth`

**AlexNet (LPIPS):**
- Link: https://drive.google.com/file/d/1FF3BBSDIA3uavmAiuMH6YFCv09Lt8jUr/view?usp=sharing
- Save as: `eval_models/weights/v0.1/alex.pth`

### 2. Create Directory Structure
```bash
mkdir checkpoints
mkdir data
mkdir eval_models/weights/v0.1
```

### 3. Download Dataset
The model uses the VITON-HD dataset. You can download it from:
https://github.com/shadow2496/VITON-HD

Place the dataset in the `./data` directory.

### 4. Test the Installation
Once all installations are complete, test with:

```bash
python test_generator.py --help
```

### 5. Run Inference
After downloading checkpoints and dataset:

```bash
python test_generator.py \
  --occlusion \
  --cuda True \
  --test_name test_run \
  --tocg_checkpoint checkpoints/condition_generator.pth \
  --gpu_ids 0 \
  --gen_checkpoint checkpoints/image_generator.pth \
  --datasetting unpaired \
  --dataroot ./data \
  --data_list test_pairs.txt
```

## Data Preparation Requirements:

### For Custom Images:
1. **OpenPose**: Extract pose keypoints
2. **Human Parsing**: Generate segmentation maps
3. **DensePose**: Generate UV maps
4. **Cloth Masking**: Remove backgrounds from clothing items
5. **Parse Agnostic**: Generate clothing-agnostic images

### File Structure Expected:
```
data/
├── image/          # Person images
├── cloth/          # Clothing items
├── cloth-mask/     # Clothing masks
├── image-parse/    # Human parsing results
├── openpose-json/  # Pose keypoints
└── densepose/      # DensePose results
```

## Troubleshooting:
- If CUDA errors occur, ensure you have compatible GPU drivers
- For memory issues, reduce batch size or use CPU mode
- Check that all checkpoint files are properly downloaded and placed in correct directories

## Quick Start with Sample Data:
1. Download the VITON-HD dataset
2. Download all model checkpoints
3. Run the test script with your data
4. Results will be saved in `./results/` 