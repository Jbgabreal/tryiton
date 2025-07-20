#!/usr/bin/env python3
"""
Compare Original vs Enhanced HR-VITON Results
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import cv2

def calculate_metrics(original, enhanced):
    """Calculate quality metrics between original and enhanced images"""
    
    # Convert to numpy arrays
    if isinstance(original, Image.Image):
        original = np.array(original)
    if isinstance(enhanced, Image.Image):
        enhanced = np.array(enhanced)
    
    # Ensure same size
    if original.shape != enhanced.shape:
        enhanced = cv2.resize(enhanced, (original.shape[1], original.shape[0]))
    
    # Convert to grayscale for some metrics
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
        enhanced_gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    else:
        original_gray = original
        enhanced_gray = enhanced
    
    # Calculate metrics
    metrics = {}
    
    # SSIM (Structural Similarity Index)
    metrics['ssim'] = ssim(original_gray, enhanced_gray)
    
    # PSNR (Peak Signal-to-Noise Ratio)
    metrics['psnr'] = psnr(original_gray, enhanced_gray)
    
    # MSE (Mean Squared Error)
    metrics['mse'] = np.mean((original_gray.astype(float) - enhanced_gray.astype(float)) ** 2)
    
    # Color difference
    if len(original.shape) == 3:
        color_diff = np.mean(np.abs(original.astype(float) - enhanced.astype(float)))
        metrics['color_diff'] = color_diff
    
    return metrics

def create_comparison_grid(original_dir, enhanced_dir, output_dir):
    """Create a comparison grid of original vs enhanced results"""
    
    # Get image files
    original_files = [f for f in os.listdir(original_dir) if f.endswith(('.png', '.jpg'))]
    enhanced_files = [f for f in os.listdir(enhanced_dir) if f.endswith(('.png', '.jpg'))]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    
    for i, (orig_file, enh_file) in enumerate(zip(original_files, enhanced_files)):
        # Load images
        original_path = os.path.join(original_dir, orig_file)
        enhanced_path = os.path.join(enhanced_dir, enh_file)
        
        original_img = Image.open(original_path)
        enhanced_img = Image.open(enhanced_path)
        
        # Calculate metrics
        metrics = calculate_metrics(original_img, enhanced_img)
        
        # Create comparison image
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original_img)
        axes[0].set_title(f'Original - {orig_file}')
        axes[0].axis('off')
        
        axes[1].imshow(enhanced_img)
        axes[1].set_title(f'Enhanced - {enh_file}')
        axes[1].axis('off')
        
        # Add metrics text
        metrics_text = f"SSIM: {metrics['ssim']:.3f}\nPSNR: {metrics['psnr']:.2f}\nMSE: {metrics['mse']:.3f}"
        if 'color_diff' in metrics:
            metrics_text += f"\nColor Diff: {metrics['color_diff']:.2f}"
        
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # Save comparison
        comparison_path = os.path.join(output_dir, f'comparison_{i:04d}.png')
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Store results
        results.append({
            'file': orig_file,
            'metrics': metrics,
            'comparison_path': comparison_path
        })
        
        print(f"Processed {orig_file}: SSIM={metrics['ssim']:.3f}, PSNR={metrics['psnr']:.2f}")
    
    return results

def generate_summary_report(results, output_dir):
    """Generate a summary report of all comparisons"""
    
    # Calculate average metrics
    avg_metrics = {}
    for key in results[0]['metrics'].keys():
        avg_metrics[key] = np.mean([r['metrics'][key] for r in results])
    
    # Create summary
    summary_path = os.path.join(output_dir, 'summary_report.txt')
    with open(summary_path, 'w') as f:
        f.write("HR-VITON Enhancement Comparison Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Average Metrics:\n")
        for metric, value in avg_metrics.items():
            f.write(f"{metric.upper()}: {value:.4f}\n")
        
        f.write(f"\nTotal images compared: {len(results)}\n")
        
        f.write("\nIndividual Results:\n")
        f.write("-" * 30 + "\n")
        for result in results:
            f.write(f"{result['file']}:\n")
            for metric, value in result['metrics'].items():
                f.write(f"  {metric.upper()}: {value:.4f}\n")
            f.write("\n")
    
    print(f"\nSummary report saved to: {summary_path}")
    print(f"Average SSIM: {avg_metrics['ssim']:.3f}")
    print(f"Average PSNR: {avg_metrics['psnr']:.2f}")

def main():
    """Main comparison function"""
    
    # Directories
    original_dir = "./output/test/test/unpaired/generator/output"
    enhanced_dir = "./output/test/enhanced"
    comparison_dir = "./output/comparisons"
    
    print("HR-VITON Results Comparison")
    print("=" * 40)
    
    # Check if directories exist
    if not os.path.exists(original_dir):
        print(f"❌ Original results directory not found: {original_dir}")
        return
    
    if not os.path.exists(enhanced_dir):
        print(f"❌ Enhanced results directory not found: {enhanced_dir}")
        print("Please run enhanced_test.py first")
        return
    
    print(f"Original results: {original_dir}")
    print(f"Enhanced results: {enhanced_dir}")
    print(f"Comparison output: {comparison_dir}")
    
    # Create comparisons
    print("\nCreating comparisons...")
    results = create_comparison_grid(original_dir, enhanced_dir, comparison_dir)
    
    # Generate summary
    print("\nGenerating summary report...")
    generate_summary_report(results, comparison_dir)
    
    print(f"\n🎉 Comparison completed!")
    print(f"Results saved in: {comparison_dir}")
    print(f"Check the summary report for detailed metrics.")

if __name__ == "__main__":
    main() 