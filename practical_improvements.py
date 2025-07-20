#!/usr/bin/env python3
"""
Practical HR-VITON Improvements
Apply enhancement techniques to improve your virtual try-on results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import os
import cv2
import argparse

def load_image(image_path):
    """Load and preprocess image"""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None
    
    image = Image.open(image_path).convert('RGB')
    return image

def enhance_output_quality(image, enhancement_level='medium'):
    """
    Apply comprehensive enhancement to HR-VITON output
    
    Args:
        image: PIL Image or numpy array
        enhancement_level: 'light', 'medium', 'strong'
    """
    
    # Convert to PIL if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    enhanced = image.copy()
    
    # Enhancement parameters based on level
    if enhancement_level == 'light':
        sharpness_factor = 1.2
        contrast_factor = 1.05
        brightness_factor = 1.02
        color_factor = 1.05
    elif enhancement_level == 'medium':
        sharpness_factor = 1.4
        contrast_factor = 1.1
        brightness_factor = 1.05
        color_factor = 1.1
    else:  # strong
        sharpness_factor = 1.6
        contrast_factor = 1.15
        brightness_factor = 1.08
        color_factor = 1.15
    
    # 1. Sharpening
    enhancer = ImageEnhance.Sharpness(enhanced)
    enhanced = enhancer.enhance(sharpness_factor)
    
    # 2. Contrast enhancement
    contrast_enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = contrast_enhancer.enhance(contrast_factor)
    
    # 3. Brightness adjustment
    brightness_enhancer = ImageEnhance.Brightness(enhanced)
    enhanced = brightness_enhancer.enhance(brightness_factor)
    
    # 4. Color saturation
    color_enhancer = ImageEnhance.Color(enhanced)
    enhanced = color_enhancer.enhance(color_factor)
    
    return enhanced

def refine_edges(image, radius=2):
    """Refine edges using unsharp mask"""
    
    # Apply unsharp mask
    enhanced = image.filter(ImageFilter.UnsharpMask(radius=radius, percent=150, threshold=3))
    
    return enhanced

def reduce_noise(image, strength='medium'):
    """Reduce noise while preserving details"""
    
    # Convert to numpy
    img_array = np.array(image)
    
    # Noise reduction parameters
    if strength == 'light':
        h = 10
    elif strength == 'medium':
        h = 15
    else:  # strong
        h = 20
    
    # Apply bilateral filter for edge-preserving noise reduction
    if len(img_array.shape) == 3:
        denoised = cv2.bilateralFilter(img_array, 9, h, h)
    else:
        denoised = img_array
    
    return Image.fromarray(denoised)

def color_correction(image, reference_image=None):
    """Correct colors to match reference or improve overall appearance"""
    
    if reference_image is None:
        # Auto color correction
        img_array = np.array(image)
        
        # Simple auto white balance
        if len(img_array.shape) == 3:
            # Calculate mean for each channel
            means = np.mean(img_array, axis=(0, 1))
            max_mean = np.max(means)
            
            # Normalize to the brightest channel
            if max_mean > 0:
                img_array = img_array * (255.0 / max_mean)
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    else:
        # Match to reference image
        return apply_style_transfer(image, reference_image)

def apply_style_transfer(source, target):
    """Apply style transfer to match target image characteristics"""
    
    # Convert to numpy arrays
    source_array = np.array(source)
    target_array = np.array(target)
    
    # Match color distribution
    if len(source_array.shape) == 3 and len(target_array.shape) == 3:
        for i in range(3):
            source_array[:,:,i] = match_histogram(source_array[:,:,i], target_array[:,:,i])
    
    return Image.fromarray(source_array)

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

def improve_clothing_fit(image, mask=None):
    """Improve clothing fit and reduce artifacts"""
    
    img_array = np.array(image)
    
    if mask is not None:
        # Use mask to focus on clothing areas
        mask_array = np.array(mask)
        if len(mask_array.shape) == 3:
            mask_array = mask_array[:,:,0]  # Use first channel
        
        # Apply smoothing only to clothing areas
        smoothed = cv2.bilateralFilter(img_array, 9, 75, 75)
        
        # Blend original and smoothed based on mask
        alpha = 0.7
        result = img_array * (1 - alpha * mask_array[:,:,np.newaxis] / 255.0) + \
                smoothed * (alpha * mask_array[:,:,np.newaxis] / 255.0)
        
        return Image.fromarray(result.astype(np.uint8))
    else:
        # Apply general smoothing
        smoothed = cv2.bilateralFilter(img_array, 9, 75, 75)
        return Image.fromarray(smoothed)

def create_comparison(original, enhanced, output_path):
    """Create side-by-side comparison"""
    
    # Ensure same size
    if original.size != enhanced.size:
        enhanced = enhanced.resize(original.size, Image.LANCZOS)
    
    # Create comparison image
    width, height = original.size
    comparison = Image.new('RGB', (width * 2, height))
    
    comparison.paste(original, (0, 0))
    comparison.paste(enhanced, (width, 0))
    
    # Add labels
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(comparison)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), "Original", fill="white", font=font)
    draw.text((width + 10, 10), "Enhanced", fill="white", font=font)
    
    comparison.save(output_path)
    return comparison

def process_hr_viton_output(input_path, output_path, enhancement_level='medium', 
                          reference_path=None, create_comparison_img=True):
    """
    Process HR-VITON output with improvements
    
    Args:
        input_path: Path to HR-VITON output image
        output_path: Path to save enhanced image
        enhancement_level: 'light', 'medium', 'strong'
        reference_path: Path to reference image for color correction
        create_comparison_img: Whether to create comparison image
    """
    
    print(f"Processing: {input_path}")
    
    # Load input image
    original_image = load_image(input_path)
    if original_image is None:
        return False
    
    # Apply enhancement pipeline
    enhanced = original_image.copy()
    
    print("Applying enhancement pipeline...")
    
    # 1. Basic quality enhancement
    enhanced = enhance_output_quality(enhanced, enhancement_level)
    
    # 2. Edge refinement
    enhanced = refine_edges(enhanced)
    
    # 3. Noise reduction
    enhanced = reduce_noise(enhanced, enhancement_level)
    
    # 4. Color correction
    if reference_path:
        reference_image = load_image(reference_path)
        if reference_image:
            enhanced = color_correction(enhanced, reference_image)
    else:
        enhanced = color_correction(enhanced)
    
    # 5. Clothing fit improvement
    enhanced = improve_clothing_fit(enhanced)
    
    # Save enhanced result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    enhanced.save(output_path)
    print(f"✅ Enhanced result saved: {output_path}")
    
    # Create comparison if requested
    if create_comparison_img:
        comparison_path = output_path.replace('.png', '_comparison.png').replace('.jpg', '_comparison.jpg')
        create_comparison(original_image, enhanced, comparison_path)
        print(f"✅ Comparison saved: {comparison_path}")
    
    return True

def batch_process_directory(input_dir, output_dir, enhancement_level='medium'):
    """Process all images in a directory"""
    
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
    
    print(f"Found {len(image_files)} images to process")
    
    for i, filename in enumerate(image_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"enhanced_{filename}")
        
        print(f"\nProcessing {i+1}/{len(image_files)}: {filename}")
        process_hr_viton_output(input_path, output_path, enhancement_level)
    
    print(f"\n🎉 Batch processing completed!")
    print(f"Enhanced results saved in: {output_dir}")

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Enhance HR-VITON outputs')
    parser.add_argument('--input', type=str, help='Input image path or directory')
    parser.add_argument('--output', type=str, help='Output path or directory')
    parser.add_argument('--level', choices=['light', 'medium', 'strong'], default='medium',
                       help='Enhancement level')
    parser.add_argument('--reference', type=str, help='Reference image for color correction')
    parser.add_argument('--batch', action='store_true', help='Process directory in batch mode')
    
    args = parser.parse_args()
    
    if args.batch:
        if not args.input or not args.output:
            print("❌ For batch mode, specify --input and --output directories")
            return
        batch_process_directory(args.input, args.output, args.level)
    else:
        if not args.input or not args.output:
            print("❌ Specify --input and --output paths")
            return
        process_hr_viton_output(args.input, args.output, args.level, args.reference)

if __name__ == "__main__":
    main() 