#!/usr/bin/env python3
"""
Mini Project Assignment 1: Smart Document Scanner & Quality Analysis System

Learning Objectives:
- Understand image sensing and acquisition mechanisms
- Apply grayscale conversion and preprocessing
- Analyze effects of sampling (resolution reduction)
- Analyze effects of quantization (bit-depth reduction)
- Visually and analytically compare image quality

This script simulates a real-world document digitization system and analyzes
how sampling and quantization affect document quality.
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common.utils import (
    load_image, save_image, convert_to_grayscale, 
    downsample_image, upsample_image, quantize_image,
    display_images, resize_image, print_section_header, compute_mse, compute_psnr
)


def task_1_setup():
    """Task 1: Project Setup and Introduction"""
    print_section_header("Task 1: Project Setup and Introduction")
    print("\nWelcome to the Document Scanning System!")
    print("\nThis system simulates a real-world document digitization workflow that:")
    print("  • Captures documents using scanners or cameras")
    print("  • Analyzes the effects of sampling (resolution) on document quality")
    print("  • Analyzes the effects of quantization (bit-depth) on document quality")
    print("  • Provides visual and analytical comparisons")
    print("\nProject folder structure:")
    print("  document_scanner/")
    print("    +-- scanner.py (this script)")
    print("    +-- outputs/")
    print("    |   +-- sampling_analysis/")
    print("    |   +-- quantization_analysis/")
    print("    |   +-- comparison_figures/")
    print("    +-- README.md")


def task_2_acquisition(image_path):
    """Task 2: Image Acquisition"""
    print_section_header("Task 2: Image Acquisition & Preprocessing")
    
    # Load image
    print(f"\nLoading image from: {image_path}")
    original_img = load_image(image_path)
    
    # Resize to standard size
    print("Resizing image to standard size (512×512)...")
    original_img = resize_image(original_img, size=(512, 512))
    
    # Convert to grayscale
    print("Converting to grayscale...")
    gray_img = convert_to_grayscale(original_img)
    
    # Display results
    display_images(
        [original_img, gray_img],
        ["Original Color", "Grayscale"],
        figsize=(10, 5)
    )
    
    return gray_img


def task_3_sampling(gray_img, output_dir):
    """Task 3: Image Sampling (Resolution Analysis)"""
    print_section_header("Task 3: Image Sampling (Resolution Analysis)")
    
    sampling_results = {}
    resolutions = [512, 256, 128]
    
    for resolution in resolutions:
        print(f"\nProcessing {resolution}×{resolution} resolution...")
        
        # Calculate downsampling factor
        factor = 512 // resolution
        
        # Downsample
        downsampled = downsample_image(gray_img, factor)
        
        # Upsample back to original size for comparison
        upsampled = upsample_image(downsampled, (512, 512))
        
        # Compute quality metrics
        mse = compute_mse(gray_img, upsampled)
        psnr = compute_psnr(gray_img, upsampled)
        
        sampling_results[resolution] = {
            'image': upsampled,
            'mse': mse,
            'psnr': psnr,
            'factor': factor
        }
        
        print(f"  MSE:  {mse:.4f}")
        print(f"  PSNR: {psnr:.4f} dB")
        print(f"  Observation: Text clarity degradation visible at {resolution}×{resolution}")
        
        # Save output
        output_file = os.path.join(output_dir, f'sampling_{resolution}x{resolution}.jpg')
        save_image(upsampled, output_file)
        print(f"  Saved to: {output_file}")
    
    # Display all sampling results
    images = [gray_img] + [sampling_results[r]['image'] for r in resolutions]
    titles = ["Original (512×512)"] + [f"Sampled {r}×{r}\nPSNR: {sampling_results[r]['psnr']:.2f} dB" 
                                        for r in resolutions]
    
    fig = display_images(images, titles, figsize=(14, 5))
    fig_path = os.path.join(output_dir, 'sampling_comparison.png')
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    print(f"\nSampling comparison figure saved to: {fig_path}")
    
    return sampling_results


def task_4_quantization(gray_img, output_dir):
    """Task 4: Image Quantization (Gray Level Reduction)"""
    print_section_header("Task 4: Image Quantization (Gray Level Reduction)")
    
    quantization_results = {}
    quantization_levels = [256, 16, 4, 2]
    bit_depths = [8, 4, 2, 1]
    
    for levels, bits in zip(quantization_levels, bit_depths):
        print(f"\nProcessing {levels} gray levels ({bits}-bit)...")
        
        # Quantize
        quantized = quantize_image(gray_img, levels)
        
        # Compute quality metrics
        mse = compute_mse(gray_img, quantized)
        psnr = compute_psnr(gray_img, quantized)
        
        quantization_results[levels] = {
            'image': quantized,
            'mse': mse,
            'psnr': psnr,
            'bits': bits
        }
        
        print(f"  MSE:  {mse:.4f}")
        print(f"  PSNR: {psnr:.4f} dB")
        print(f"  Observation: Posterization artifacts visible at {bits}-bit")
        
        # Save output
        output_file = os.path.join(output_dir, f'quantized_{levels}_levels_{bits}bit.jpg')
        save_image(quantized, output_file)
        print(f"  Saved to: {output_file}")
    
    # Display all quantization results
    images = [gray_img] + [quantization_results[l]['image'] for l in quantization_levels[1:]]
    titles = ["Original (8-bit, 256 levels)"] + \
             [f"Quantized ({quantization_results[l]['bits']}-bit, {l} levels)\nPSNR: {quantization_results[l]['psnr']:.2f} dB" 
              for l in quantization_levels[1:]]
    
    fig = display_images(images, titles, figsize=(14, 5))
    fig_path = os.path.join(output_dir, 'quantization_comparison.png')
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    print(f"\nQuantization comparison figure saved to: {fig_path}")
    
    return quantization_results


def task_5_analysis(gray_img, sampling_results, quantization_results):
    """Task 5: Quality Observation & Analysis"""
    print_section_header("Task 5: Quality Observation & Analysis")
    
    print("\n" + "="*70)
    print("SAMPLING ANALYSIS - Loss of Fine Details")
    print("="*70)
    
    for resolution in [512, 256, 128]:
        result = sampling_results[resolution]
        print(f"\n{resolution}×{resolution} Resolution:")
        print(f"  • MSE:  {result['mse']:.4f}")
        print(f"  • PSNR: {result['psnr']:.4f} dB")
        
        if resolution == 512:
            print(f"  • Assessment: HIGH QUALITY - Original resolution maintained")
        elif resolution == 256:
            print(f"  • Assessment: MEDIUM QUALITY - Fine text details slightly blurred")
            print(f"  • OCR Suitability: GOOD - Text still recognizable")
        else:  # 128
            print(f"  • Assessment: LOW QUALITY - Significant detail loss")
            print(f"  • OCR Suitability: POOR - Text becomes fragmented and illegible")
    
    print("\n" + "="*70)
    print("QUANTIZATION ANALYSIS - Bit-Depth Reduction")
    print("="*70)
    
    for levels in [256, 16, 4, 2]:
        result = quantization_results[levels]
        print(f"\n{result['bits']}-bit, {levels} Gray Levels:")
        print(f"  • MSE:  {result['mse']:.4f}")
        print(f"  • PSNR: {result['psnr']:.4f} dB")
        
        if levels == 256:
            print(f"  • Assessment: NO DEGRADATION - Original 8-bit quality")
        elif levels == 16:
            print(f"  • Assessment: ACCEPTABLE - Minor posterization visible")
            print(f"  • OCR Suitability: ACCEPTABLE - Some character features lost")
        elif levels == 4:
            print(f"  • Assessment: POOR - Severe posterization")
            print(f"  • OCR Suitability: DIFFICULT - Text outlines distorted")
        else:  # 2
            print(f"  • Assessment: VERY POOR - Binary image, extreme loss")
            print(f"  • OCR Suitability: UNUSABLE - Text becomes unrecognizable")
    
    print("\n" + "="*70)
    print("OVERALL CONCLUSIONS")
    print("="*70)
    print("\n1. SAMPLING EFFECTS:")
    print("   • Resolution of 512×512 maintains document legibility")
    print("   • Resolution of 256×256 acceptable for most OCR systems")
    print("   • Resolution of 128×128 unsuitable for quality document digitization")
    print("\n2. QUANTIZATION EFFECTS:")
    print("   • 8-bit (256 levels) essential for maintaining text clarity")
    print("   • 4-bit (16 levels) causes visible artifacts but text still readable")
    print("   • 2-bit (4 levels) severely compromises OCR accuracy")
    print("\n3. RECOMMENDATIONS FOR DOCUMENT SCANNING:")
    print("   • Minimum resolution: 300 DPI (~512x512 for letter-size)")
    print("   • Bit-depth: 8-bit grayscale or higher")
    print("   • Use lossless compression (PNG, TIFF) for archival")


def main():
    parser = argparse.ArgumentParser(description='Smart Document Scanner - Assignment 1')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to input document image')
    parser.add_argument('--output', '-o', type=str, default='outputs/asmt1',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Task 1: Setup
    task_1_setup()
    
    # Determine input image path
    if args.input is None:
        # Try to find a sample image
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sample_paths = [
            os.path.join(base_dir, '../../assets/asmt1/sample_1.jpg'),
            os.path.join(base_dir, '../../assets/asmt1/sample_1.jpg'),
        ]
        
        input_path = None
        for path in sample_paths:
            if os.path.exists(path):
                input_path = path
                break
        
        if input_path is None:
            print("\nError: No input image found. Please provide --input argument.")
            sys.exit(1)
    else:
        input_path = args.input
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Task 2: Acquisition
    gray_img = task_2_acquisition(input_path)
    
    # Task 3: Sampling Analysis
    sampling_results = task_3_sampling(gray_img, args.output)
    
    # Task 4: Quantization Analysis
    quantization_results = task_4_quantization(gray_img, args.output)
    
    # Task 5: Analysis
    task_5_analysis(gray_img, sampling_results, quantization_results)
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"\nAll outputs saved to: {os.path.abspath(args.output)}")
    print("\nGenerated files:")
    print("  • sampling_*.jpg - Individual sampling results")
    print("  • quantized_*.jpg - Individual quantization results")
    print("  • sampling_comparison.png - Sampling analysis figure")
    print("  • quantization_comparison.png - Quantization analysis figure")
    
    plt.close('all')


if __name__ == '__main__':
    main()
