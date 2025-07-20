#!/usr/bin/env python3
"""
Test HR-VITON model loading without requiring full dataset
"""

import torch
import torch.nn as nn
import os
import sys

def test_model_loading():
    """Test if the models can be loaded correctly"""
    
    print("HR-VITON Model Loading Test")
    print("=" * 40)
    
    # Check if required files exist
    required_files = [
        'checkpoints/condition_generator.pth',
        'checkpoints/condition_discriminator.pth',
        'checkpoints/image_generator.pth',
        'eval_models/weights/v0.1/alex.pth'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            print(f"❌ Missing: {file_path}")
            return False
        else:
            print(f"✅ Found: {file_path}")
    
    print("\nTesting model loading...")
    
    try:
        # Import the model classes
        from networks import ConditionGenerator
        from network_generator import SPADEGenerator
        
        print("✅ Model classes imported successfully")
        
        # Test condition generator loading
        print("\nTesting condition generator...")
        input1_nc = 4  # cloth + cloth-mask
        input2_nc = 13 + 3  # parse_agnostic + densepose
        output_nc = 13
        
        # Create a simple options object
        class Opt:
            def __init__(self):
                self.semantic_nc = 13
                self.output_nc = 13
                self.cuda = False
                self.warp_feature = "T1"
                self.out_layer = "relu"
                self.num_upsampling_layers = "most"
                self.norm_G = "spectralaliasinstance"
                self.ngf = 64
                self.init_type = "xavier"
                self.init_variance = 0.02
                self.fine_width = 768
                self.fine_height = 1024
                self.gen_semantic_nc = 7
        
        opt = Opt()
        
        tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
        print("✅ Condition generator created successfully")
        
        # Test loading checkpoint
        from networks import load_checkpoint
        load_checkpoint(tocg, 'checkpoints/condition_generator.pth', opt)
        print("✅ Condition generator checkpoint loaded successfully")
        
        # Test image generator loading
        print("\nTesting image generator...")
        opt.semantic_nc = 7
        generator = SPADEGenerator(opt, 3+3+3)
        print("✅ Image generator created successfully")
        
        # Test loading checkpoint
        from test_generator import load_checkpoint_G
        load_checkpoint_G(generator, 'checkpoints/image_generator.pth', opt)
        print("✅ Image generator checkpoint loaded successfully")
        
        # Test with dummy data
        print("\nTesting with dummy data...")
        batch_size = 1
        dummy_input1 = torch.randn(batch_size, input1_nc, 256, 192)
        dummy_input2 = torch.randn(batch_size, input2_nc, 256, 192)
        
        # Test condition generator forward pass
        with torch.no_grad():
            flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt, dummy_input1, dummy_input2)
            print("✅ Condition generator forward pass successful")
            
            # Test image generator forward pass
            dummy_parse = torch.randn(batch_size, 7, 1024, 768)
            dummy_agnostic = torch.randn(batch_size, 3, 1024, 768)
            dummy_densepose = torch.randn(batch_size, 3, 1024, 768)
            dummy_warped_cloth = torch.randn(batch_size, 3, 1024, 768)
            
            generator_input = torch.cat((dummy_agnostic, dummy_densepose, dummy_warped_cloth), dim=1)
            output = generator(generator_input, dummy_parse)
            print("✅ Image generator forward pass successful")
            
            print(f"✅ Output shape: {output.shape}")
        
        print("\n🎉 All model tests passed!")
        print("The HR-VITON models are working correctly.")
        print("\nNote: To run with real data, you need the complete dataset structure.")
        print("See demo_custom_images.py for data requirements.")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    success = test_model_loading()
    
    if success:
        print("\n✅ Model loading test completed successfully!")
        print("Your HR-VITON setup is working correctly.")
    else:
        print("\n❌ Model loading test failed.")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main() 