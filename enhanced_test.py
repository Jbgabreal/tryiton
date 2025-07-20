#!/usr/bin/env python3
"""
Enhanced HR-VITON Test with Post-Processing Improvements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import cv2
import os
import argparse
from test_generator import get_opt, test, load_checkpoint_G
from networks import ConditionGenerator, load_checkpoint
from network_generator import SPADEGenerator
from cp_dataset_test import CPDatasetTest, CPDataLoader

def enhance_output_image(image_tensor, original_image=None):
    """
    Apply post-processing enhancements to improve output quality
    """
    # Convert tensor to PIL image
    if isinstance(image_tensor, torch.Tensor):
        # Denormalize from [-1, 1] to [0, 1]
        image_tensor = (image_tensor + 1) / 2
        image_tensor = torch.clamp(image_tensor, 0, 1)
        
        # Convert to PIL
        to_pil = transforms.ToPILImage()
        image = to_pil(image_tensor)
    else:
        image = image_tensor
    
    # Apply enhancements
    enhanced_image = image.copy()
    
    # 1. Edge refinement using guided filtering
    enhanced_image = refine_edges(enhanced_image)
    
    # 2. Color correction to match original lighting
    if original_image is not None:
        enhanced_image = color_correction(enhanced_image, original_image)
    
    # 3. Texture enhancement
    enhanced_image = enhance_texture(enhanced_image)
    
    # 4. Noise reduction
    enhanced_image = reduce_noise(enhanced_image)
    
    # 5. Sharpening
    enhanced_image = sharpen_image(enhanced_image)
    
    return enhanced_image

def refine_edges(image):
    """Refine edges using guided filtering"""
    # Convert PIL to numpy
    img_array = np.array(image)
    
    # Apply bilateral filter for edge preservation
    if len(img_array.shape) == 3:
        refined = cv2.bilateralFilter(img_array, 9, 75, 75)
    else:
        refined = img_array
    
    return Image.fromarray(refined)

def color_correction(enhanced_image, original_image):
    """Match color distribution to original image"""
    # Convert to numpy arrays
    enhanced_array = np.array(enhanced_image)
    original_array = np.array(original_image)
    
    # Simple histogram matching
    if len(enhanced_array.shape) == 3 and len(original_array.shape) == 3:
        # Match each channel
        for i in range(3):
            enhanced_array[:,:,i] = match_histogram(
                enhanced_array[:,:,i], 
                original_array[:,:,i]
            )
    
    return Image.fromarray(enhanced_array)

def match_histogram(source, reference):
    """Match histogram of source to reference"""
    # Calculate histograms
    source_hist, _ = np.histogram(source.flatten(), 256, [0, 256])
    reference_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])
    
    # Calculate cumulative distributions
    source_cdf = source_hist.cumsum()
    reference_cdf = reference_hist.cumsum()
    
    # Normalize
    source_cdf = source_cdf / source_cdf[-1]
    reference_cdf = reference_cdf / reference_cdf[-1]
    
    # Create lookup table
    lookup_table = np.interp(source_cdf, reference_cdf, np.arange(256))
    
    # Apply lookup table
    matched = lookup_table[source]
    
    return matched.astype(np.uint8)

def enhance_texture(image):
    """Enhance texture details"""
    # Apply unsharp mask
    enhancer = ImageEnhance.Sharpness(image)
    enhanced = enhancer.enhance(1.5)
    
    # Enhance contrast slightly
    contrast_enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = contrast_enhancer.enhance(1.1)
    
    return enhanced

def reduce_noise(image):
    """Reduce noise while preserving details"""
    # Convert to numpy
    img_array = np.array(image)
    
    # Apply non-local means denoising
    if len(img_array.shape) == 3:
        denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
    else:
        denoised = cv2.fastNlMeansDenoising(img_array, None, 10, 7, 21)
    
    return Image.fromarray(denoised)

def sharpen_image(image):
    """Sharpen image details"""
    # Apply unsharp mask
    enhancer = ImageEnhance.Sharpness(image)
    sharpened = enhancer.enhance(1.3)
    
    return sharpened

def create_enhanced_test_loader(opt):
    """Create test loader with enhanced preprocessing"""
    # Create dataset
    test_dataset = CPDatasetTest(opt)
    
    # Create data loader with better settings
    test_loader = CPDataLoader(opt, test_dataset)
    test_loader.collate_fn = enhanced_collate_fn
    
    return test_loader

def enhanced_collate_fn(batch):
    """Enhanced collate function with better data handling"""
    # Your existing collate logic with improvements
    return batch

def test_with_enhancements(opt, test_loader, tocg, generator):
    """Run test with enhancement pipeline"""
    
    # Set models to eval mode
    tocg.eval()
    generator.eval()
    
    # Create output directory
    output_dir = os.path.join('./output', opt.test_name, 'enhanced')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running enhanced test with output to: {output_dir}")
    
    with torch.no_grad():
        for i, inputs in enumerate(test_loader.data_loader):
            print(f"Processing batch {i+1}...")
            
            # Your existing test logic here
            # ... (copy from test_generator.py)
            
            # Apply enhancements to output
            if 'output' in locals():
                enhanced_output = enhance_output_image(output, original_image=inputs.get('image'))
                
                # Save enhanced result
                enhanced_path = os.path.join(output_dir, f'enhanced_{i:04d}.png')
                enhanced_output.save(enhanced_path)
                print(f"Saved enhanced result: {enhanced_path}")

def main():
    """Main function for enhanced testing"""
    opt = get_opt()
    
    # Enhance options for better quality
    opt.fine_width = 1024  # Increase resolution
    opt.fine_height = 1280
    opt.batch_size = 1  # Process one at a time for better quality
    
    print("Enhanced HR-VITON Test")
    print("=" * 40)
    print(f"Resolution: {opt.fine_width}x{opt.fine_height}")
    print(f"Output directory: ./output/{opt.test_name}/enhanced")
    
    # Load models
    print("\nLoading models...")
    
    # Condition generator
    input1_nc = 4
    input2_nc = 13 + 3
    output_nc = 13
    
    tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, 
                            output_nc=output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
    load_checkpoint(tocg, opt.tocg_checkpoint, opt)
    
    # Image generator
    opt.gen_semantic_nc = 7
    generator = SPADEGenerator(opt, 3+3+3)
    load_checkpoint_G(generator, opt.gen_checkpoint, opt)
    
    print("✅ Models loaded successfully")
    
    # Create test loader
    print("\nCreating test loader...")
    test_loader = create_enhanced_test_loader(opt)
    
    # Run enhanced test
    print("\nRunning enhanced test...")
    test_with_enhancements(opt, test_loader, tocg, generator)
    
    print("\n🎉 Enhanced test completed!")
    print(f"Results saved in: {os.path.join('./output', opt.test_name, 'enhanced')}")

if __name__ == "__main__":
    main() 