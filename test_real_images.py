#!/usr/bin/env python3
"""
Test HR-VITON on Real Test Data
Run virtual try-on on actual person and clothing images
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from test_generator import get_opt, load_checkpoint_G
from networks import ConditionGenerator, load_checkpoint
from network_generator import SPADEGenerator
from cp_dataset_test import CPDatasetTest, CPDataLoader

def create_test_pairs_file():
    """Create a test pairs file with available images"""
    
    print("Creating test pairs file...")
    
    # Get available images
    image_dir = 'data/test/image'
    cloth_dir = 'data/test/cloth'
    
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    cloth_files = [f for f in os.listdir(cloth_dir) if f.endswith('.jpg')]
    
    # Take first 10 pairs for testing
    test_pairs = []
    for i in range(min(10, len(image_files), len(cloth_files))):
        test_pairs.append(f"{image_files[i]} {cloth_files[i]}")
    
    # Write test pairs file
    with open('data/test_pairs_available.txt', 'w') as f:
        for pair in test_pairs:
            f.write(pair + '\n')
    
    print(f"✅ Created test pairs file with {len(test_pairs)} pairs")
    return test_pairs

def test_with_real_images():
    """Test HR-VITON with real test images"""
    
    print("HR-VITON Test with Real Images")
    print("=" * 50)
    
    # Check if models exist
    model_files = [
        'checkpoints/condition_generator.pth',
        'checkpoints/image_generator.pth'
    ]
    
    for file_path in model_files:
        if not os.path.exists(file_path):
            print(f"❌ Missing model file: {file_path}")
            return False
        else:
            print(f"✅ Found: {file_path}")
    
    # Create test pairs file
    test_pairs = create_test_pairs_file()
    
    # Set up options for real data
    opt = get_opt()
    opt.dataroot = './data'
    opt.datamode = 'test'
    opt.data_list = 'test_pairs_available.txt'
    opt.fine_width = 768
    opt.fine_height = 1024
    opt.cuda = False
    opt.test_name = 'real_test'
    
    print(f"\nTest configuration:")
    print(f"Data root: {opt.dataroot}")
    print(f"Data mode: {opt.datamode}")
    print(f"Data list: {opt.data_list}")
    print(f"Resolution: {opt.fine_width}x{opt.fine_height}")
    print(f"Test pairs: {len(test_pairs)}")
    
    # Load models
    print("\nLoading models...")
    
    # Condition generator
    input1_nc = 4
    input2_nc = 13 + 3
    output_nc = 13
    
    tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, 
                            output_nc=output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
    load_checkpoint(tocg, 'checkpoints/condition_generator.pth', opt)
    print("✅ Condition generator loaded")
    
    # Image generator
    opt.gen_semantic_nc = 7
    generator = SPADEGenerator(opt, 3+3+3)
    load_checkpoint_G(generator, 'checkpoints/image_generator.pth', opt)
    print("✅ Image generator loaded")
    
    # Create dataset and loader
    print("\nCreating dataset and loader...")
    
    try:
        test_dataset = CPDatasetTest(opt)
        test_loader = CPDataLoader(opt, test_dataset)
        print("✅ Dataset and loader created successfully")
        
        # Run inference
        print("\nRunning inference on real images...")
        
        # Set models to eval mode
        tocg.eval()
        generator.eval()
        
        # Create output directory
        output_dir = './output/real_test'
        os.makedirs(output_dir, exist_ok=True)
        
        with torch.no_grad():
            for i, inputs in enumerate(test_loader.data_loader):
                print(f"Processing pair {i+1}/{len(test_pairs)}...")
                
                # Save input images for reference
                if 'image' in inputs:
                    input_img = inputs['image']
                    if isinstance(input_img, torch.Tensor):
                        # Convert tensor to PIL (remove batch dimension)
                        input_img = input_img[0]  # Take first batch
                        input_img = (input_img + 1) / 2  # Denormalize
                        input_img = torch.clamp(input_img, 0, 1)
                        input_img = transforms.ToPILImage()(input_img)
                    
                    input_img.save(os.path.join(output_dir, f'person_{i+1:02d}.png'))
                    print(f"  ✅ Saved person image: person_{i+1:02d}.png")
                
                if 'cloth' in inputs:
                    cloth_img = inputs['cloth']['unpaired']
                    if isinstance(cloth_img, torch.Tensor):
                        cloth_img = cloth_img[0]  # Take first batch
                        cloth_img = (cloth_img + 1) / 2
                        cloth_img = torch.clamp(cloth_img, 0, 1)
                        cloth_img = transforms.ToPILImage()(cloth_img)
                    
                    cloth_img.save(os.path.join(output_dir, f'clothing_{i+1:02d}.png'))
                    print(f"  ✅ Saved clothing image: clothing_{i+1:02d}.png")
                
                # Process through HR-VITON pipeline
                try:
                    # Extract inputs
                    pose_map = inputs['pose']
                    pre_clothes_mask = inputs['cloth_mask']['unpaired']
                    label = inputs['parse']
                    parse_agnostic = inputs['parse_agnostic']
                    agnostic = inputs['agnostic']
                    clothes = inputs['cloth']['unpaired']
                    densepose = inputs['densepose']
                    im = inputs['image']
                    input_label, input_parse_agnostic = label, parse_agnostic
                    pre_clothes_mask = torch.FloatTensor((pre_clothes_mask.detach().cpu().numpy() > 0.5).astype(float))
                    
                    # Downsample for processing
                    pose_map_down = F.interpolate(pose_map, size=(256, 192), mode='bilinear')
                    pre_clothes_mask_down = F.interpolate(pre_clothes_mask, size=(256, 192), mode='nearest')
                    input_label_down = F.interpolate(input_label, size=(256, 192), mode='bilinear')
                    input_parse_agnostic_down = F.interpolate(input_parse_agnostic, size=(256, 192), mode='nearest')
                    agnostic_down = F.interpolate(agnostic, size=(256, 192), mode='nearest')
                    clothes_down = F.interpolate(clothes, size=(256, 192), mode='bilinear')
                    densepose_down = F.interpolate(densepose, size=(256, 192), mode='bilinear')
                    
                    # Multi-task inputs
                    input1 = torch.cat([clothes_down, pre_clothes_mask_down], 1)
                    input2 = torch.cat([input_parse_agnostic_down, densepose_down], 1)
                    
                    # Forward through condition generator
                    flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt, input1, input2)
                    
                    # Process for image generator
                    warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(float))
                    
                    # Cloth mask composition
                    cloth_mask = torch.ones_like(fake_segmap)
                    cloth_mask[:,3:4, :, :] = warped_clothmask_paired
                    fake_segmap = fake_segmap * cloth_mask
                    
                    # Make generator input parse map
                    fake_parse_gauss = F.interpolate(fake_segmap, size=(opt.fine_height, opt.fine_width), mode='bilinear')
                    fake_parse = fake_parse_gauss.argmax(dim=1)[:, None]
                    
                    old_parse = torch.FloatTensor(fake_parse.size(0), 13, opt.fine_height, opt.fine_width).zero_()
                    old_parse.scatter_(1, fake_parse, 1.0)
                    
                    # Create parse map for generator
                    labels = {
                        0:  ['background',  [0]],
                        1:  ['paste',       [2, 4, 7, 8, 9, 10, 11]],
                        2:  ['upper',       [3]],
                        3:  ['hair',        [1]],
                        4:  ['left_arm',    [5]],
                        5:  ['right_arm',   [6]],
                        6:  ['noise',       [12]]
                    }
                    
                    parse = torch.FloatTensor(fake_parse.size(0), 7, opt.fine_height, opt.fine_width).zero_()
                    for i in range(len(labels)):
                        for label in labels[i][1]:
                            parse[:, i] += old_parse[:, label]
                    
                    # Warp cloth
                    N, _, iH, iW = clothes.shape
                    flow = F.interpolate(flow_list[-1].permute(0, 3, 1, 2), size=(iH, iW), mode='bilinear').permute(0, 2, 3, 1)
                    flow_norm = torch.cat([flow[:, :, :, 0:1] / ((96 - 1.0) / 2.0), flow[:, :, :, 1:2] / ((128 - 1.0) / 2.0)], 3)
                    
                    # Create grid for warping
                    grid = torch.stack(torch.meshgrid(torch.arange(iW), torch.arange(iH), indexing='xy'), dim=-1).float()
                    grid = grid.unsqueeze(0).repeat(N, 1, 1, 1)
                    warped_grid = grid + flow_norm
                    warped_cloth = F.grid_sample(clothes, warped_grid, padding_mode='border')
                    
                    # Generate final output
                    output = generator(torch.cat((agnostic, densepose, warped_cloth), dim=1), parse)
                    
                    # Save result
                    output_img = output[0]  # Take first batch
                    output_img = (output_img + 1) / 2  # Denormalize
                    output_img = torch.clamp(output_img, 0, 1)
                    
                    # Convert to PIL image
                    to_pil = transforms.ToPILImage()
                    result_img = to_pil(output_img)
                    
                    # Save result
                    result_path = os.path.join(output_dir, f'result_{i+1:02d}.png')
                    result_img.save(result_path)
                    print(f"  ✅ Generated virtual try-on result: result_{i+1:02d}.png")
                    
                except Exception as e:
                    print(f"  ❌ Error processing pair {i+1}: {e}")
                    continue
                
                # Process all 10 pairs
                if i >= 9:  # Process 10 pairs (0-9)
                    break
        
        print(f"\n🎉 Real image test completed!")
        print(f"Results saved in: {output_dir}")
        print("Check the output directory to see the virtual try-on results.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_comparison_grid():
    """Create a comparison grid of all results"""
    
    output_dir = './output/real_test'
    if not os.path.exists(output_dir):
        print("❌ Output directory not found")
        return
    
    # Get all result files
    result_files = [f for f in os.listdir(output_dir) if f.startswith('result_')]
    person_files = [f for f in os.listdir(output_dir) if f.startswith('person_')]
    clothing_files = [f for f in os.listdir(output_dir) if f.startswith('clothing_')]
    
    if not result_files:
        print("❌ No result files found")
        return
    
    print(f"\nCreating comparison grid...")
    print(f"Found {len(result_files)} results")
    
    # Create comparison grid
    from PIL import ImageDraw, ImageFont
    
    # Load images
    images = []
    for i in range(len(result_files)):
        person_img = Image.open(os.path.join(output_dir, person_files[i]))
        clothing_img = Image.open(os.path.join(output_dir, clothing_files[i]))
        result_img = Image.open(os.path.join(output_dir, result_files[i]))
        
        # Resize to same size
        size = (256, 256)
        person_resized = person_img.resize(size, Image.LANCZOS)
        clothing_resized = clothing_img.resize(size, Image.LANCZOS)
        result_resized = result_img.resize(size, Image.LANCZOS)
        
        images.append((person_resized, clothing_resized, result_resized))
    
    # Create grid
    cols = 3
    rows = len(images)
    grid_width = size[0] * cols
    grid_height = size[1] * rows
    
    grid = Image.new('RGB', (grid_width, grid_height))
    
    # Add images to grid
    for i, (person, clothing, result) in enumerate(images):
        row = i
        col = 0
        
        x = col * size[0]
        y = row * size[1]
        
        grid.paste(person, (x, y))
        grid.paste(clothing, (x + size[0], y))
        grid.paste(result, (x + 2 * size[0], y))
    
    # Add labels
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Add column labels
    draw.text((10, 10), "Person", fill="white", font=font)
    draw.text((size[0] + 10, 10), "Clothing", fill="white", font=font)
    draw.text((2 * size[0] + 10, 10), "Result", fill="white", font=font)
    
    # Save grid
    grid.save(os.path.join(output_dir, 'comparison_grid.png'))
    print(f"✅ Comparison grid saved: comparison_grid.png")

def main():
    """Main function"""
    
    success = test_with_real_images()
    
    if success:
        create_comparison_grid()
        
        print("\n" + "=" * 60)
        print("✅ REAL IMAGE TEST COMPLETED!")
        print("Your HR-VITON model has processed real test images.")
        print("Check ./output/real_test/ for the results.")
        print("=" * 60)
    else:
        print("\n❌ Test failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 