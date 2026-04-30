#!/usr/bin/env python3
"""
Mini Project Assignment 2: Image Restoration for Surveillance Camera Systems

Learning Objectives:
- Understand real-world image noise sources
- Implement noise models in Python
- Apply spatial filtering for image restoration
- Evaluate restoration quality using objective metrics

This project simulates real-world surveillance noise and restores image quality
using classical filtering techniques.
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
    load_image, save_image, convert_to_grayscale, resize_image,
    add_gaussian_noise, add_salt_pepper_noise,
    apply_gaussian_blur, apply_median_filter, apply_mean_filter,
    compute_mse, compute_psnr, display_images,
    print_section_header, print_metrics
)


def task_1_selection_preprocessing(image_path):
    """Task 1: Image Selection and Preprocessing"""
    print_section_header("Task 1: Image Selection and Preprocessing")
    
    print(f"\nLoading surveillance-style image from: {image_path}")
    image = load_image(image_path)
    
    print("Resizing to standard size (512×512)...")
    image = resize_image(image, size=(512, 512))
    
    print("Converting to grayscale...")
    gray_image = convert_to_grayscale(image)
    
    print("\nImage loaded and preprocessed successfully!")
    print(f"  • Dimensions: {gray_image.shape}")
    print(f"  • Data type: {gray_image.dtype}")
    print(f"  • Value range: [{gray_image.min()}, {gray_image.max()}]")
    
    display_images([gray_image], ["Original Image"], figsize=(8, 6))
    
    return gray_image


def task_2_noise_modeling(original_image, output_dir):
    """Task 2: Noise Modeling"""
    print_section_header("Task 2: Noise Modeling")
    
    noise_results = {}
    
    # Gaussian Noise (sensor noise)
    print("\nSimulating Gaussian noise (sensor noise)...")
    print("  • Mean: 0")
    print("  • Standard Deviation: 25")
    gaussian_noisy = add_gaussian_noise(original_image, mean=0, std=25)
    
    mse_g = compute_mse(original_image, gaussian_noisy)
    psnr_g = compute_psnr(original_image, gaussian_noisy)
    print(f"  • MSE:  {mse_g:.4f}")
    print(f"  • PSNR: {psnr_g:.4f} dB")
    
    noise_results['gaussian'] = {
        'image': gaussian_noisy,
        'mse': mse_g,
        'psnr': psnr_g,
        'description': 'Gaussian Noise (sensor noise from low-light conditions)'
    }
    
    save_image(gaussian_noisy, os.path.join(output_dir, 'noisy_gaussian.jpg'))
    
    # Salt-and-Pepper Noise (transmission errors)
    print("\nSimulating Salt-and-Pepper noise (transmission errors)...")
    print("  • Probability: 5%")
    sp_noisy = add_salt_pepper_noise(original_image, prob=0.05)
    
    mse_sp = compute_mse(original_image, sp_noisy)
    psnr_sp = compute_psnr(original_image, sp_noisy)
    print(f"  • MSE:  {mse_sp:.4f}")
    print(f"  • PSNR: {psnr_sp:.4f} dB")
    
    noise_results['salt_pepper'] = {
        'image': sp_noisy,
        'mse': mse_sp,
        'psnr': psnr_sp,
        'description': 'Salt-and-Pepper Noise (transmission/compression errors)'
    }
    
    save_image(sp_noisy, os.path.join(output_dir, 'noisy_salt_pepper.jpg'))
    
    # Display noisy images
    images = [original_image, gaussian_noisy, sp_noisy]
    titles = [
        "Original Image",
        f"Gaussian Noise\nPSNR: {psnr_g:.2f} dB",
        f"Salt-and-Pepper Noise\nPSNR: {psnr_sp:.2f} dB"
    ]
    fig = display_images(images, titles, figsize=(14, 5))
    fig_path = os.path.join(output_dir, 'noisy_images.png')
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    print(f"\nNoisy images figure saved to: {fig_path}")
    
    return noise_results


def task_3_restoration(original_image, noise_results, output_dir):
    """Task 3: Image Restoration Techniques"""
    print_section_header("Task 3: Image Restoration Techniques")
    
    restoration_results = {}
    
    filters = {
        'mean': apply_mean_filter,
        'median': apply_median_filter,
        'gaussian': apply_gaussian_blur
    }
    
    filter_params = {
        'mean': {'kernel_size': 5},
        'median': {'kernel_size': 5},
        'gaussian': {'kernel_size': (5, 5), 'sigma': 1.0}
    }
    
    for noise_type, noise_data in noise_results.items():
        print(f"\n{'='*70}")
        print(f"Restoring {noise_type.upper()} noise")
        print(f"{'='*70}")
        
        noisy_image = noise_data['image']
        restoration_results[noise_type] = {}
        
        for filter_name, filter_func in filters.items():
            print(f"\nApplying {filter_name.upper()} filter...")
            
            # Apply filter
            kwargs = filter_params[filter_name]
            restored = filter_func(noisy_image, **kwargs)
            
            # Compute metrics
            mse = compute_mse(original_image, restored)
            psnr = compute_psnr(original_image, restored)
            
            restoration_results[noise_type][filter_name] = {
                'image': restored,
                'mse': mse,
                'psnr': psnr
            }
            
            print(f"  • MSE:  {mse:.4f}")
            print(f"  • PSNR: {psnr:.4f} dB")
            print(f"  • Assessment: Noise suppression {'excellent' if psnr > 25 else 'good' if psnr > 20 else 'moderate'}")
            
            # Save result
            filename = f'restored_{noise_type}_{filter_name}.jpg'
            save_image(restored, os.path.join(output_dir, filename))
        
        # Display restoration results for this noise type
        images = [original_image, noisy_image] + \
                 [restoration_results[noise_type][f]['image'] for f in filters.keys()]
        
        titles = ["Original", f"Noisy\n({noise_type})"] + \
                 [f"{f.upper()} Filter\nPSNR: {restoration_results[noise_type][f]['psnr']:.2f} dB" 
                  for f in filters.keys()]
        
        fig = display_images(images, titles, figsize=(14, 5))
        fig_path = os.path.join(output_dir, f'restoration_{noise_type}.png')
        plt.savefig(fig_path, dpi=100, bbox_inches='tight')
        print(f"\nRestoration figure saved to: {fig_path}")
    
    return restoration_results


def task_4_performance_evaluation(original_image, restoration_results, noise_results):
    """Task 4: Performance Evaluation"""
    print_section_header("Task 4: Performance Evaluation")
    
    filters = ['mean', 'median', 'gaussian']
    
    for noise_type in noise_results.keys():
        print(f"\n{'='*70}")
        print(f"Performance Comparison - {noise_type.upper()} Noise")
        print(f"{'='*70}")
        print(f"\n{'Filter':<15} {'MSE':<12} {'PSNR (dB)':<15} {'Quality':<15}")
        print("-" * 60)
        
        best_psnr = -1
        best_filter = None
        
        for filter_name in filters:
            metrics = restoration_results[noise_type][filter_name]
            mse = metrics['mse']
            psnr = metrics['psnr']
            
            # Quality assessment
            if psnr > 30:
                quality = "Excellent"
            elif psnr > 25:
                quality = "Very Good"
            elif psnr > 20:
                quality = "Good"
            elif psnr > 15:
                quality = "Fair"
            else:
                quality = "Poor"
            
            print(f"{filter_name:<15} {mse:<12.4f} {psnr:<15.4f} {quality:<15}")
            
            if psnr > best_psnr:
                best_psnr = psnr
                best_filter = filter_name
        
        print(f"\nBest filter for {noise_type}: {best_filter.upper()} (PSNR: {best_psnr:.4f} dB)")


def task_5_analytical_discussion(original_image, restoration_results, noise_results):
    """Task 5: Analytical Discussion"""
    print_section_header("Task 5: Analytical Discussion")
    
    print("\n" + "="*70)
    print("FILTER CHARACTERISTICS & SUITABILITY")
    print("="*70)
    
    print("\n1. MEAN FILTER (Box Filter):")
    print("   • Method: Replaces each pixel with average of neighborhood")
    print("   • Advantages: Simple, removes additive noise uniformly")
    print("   • Disadvantages: Blurs edges, destroys fine details")
    print("   • Best for: Light to moderate Gaussian noise")
    
    print("\n2. MEDIAN FILTER:")
    print("   • Method: Replaces each pixel with median of neighborhood")
    print("   • Advantages: Excellent for salt-and-pepper noise")
    print("   • Disadvantages: Computationally more expensive")
    print("   • Best for: Impulse noise (salt-and-pepper)")
    
    print("\n3. GAUSSIAN FILTER:")
    print("   • Method: Weighted averaging with Gaussian kernel")
    print("   • Advantages: Smooth transition, natural blur, good for Gaussian noise")
    print("   • Disadvantages: Also blurs important features")
    print("   • Best for: Heavy Gaussian noise, smoothing")
    
    print("\n" + "="*70)
    print("NOISE-SPECIFIC RESTORATION RECOMMENDATIONS")
    print("="*70)
    
    print("\nGAUSSIAN NOISE (Sensor/Low-Light Noise):")
    print("  • Root Cause: Random thermal noise in camera sensor")
    print("  • Characteristics: Continuous distribution, affects all pixels")
    best_gaussian = max(
        restoration_results.get('gaussian', {}).items(),
        key=lambda x: x[1]['psnr'],
        default=('N/A', {})
    )
    if best_gaussian[0] != 'N/A':
        print(f"  • Recommended Filter: {best_gaussian[0].upper()}")
        print(f"  • Expected PSNR: {best_gaussian[1]['psnr']:.2f} dB")
    
    print("\nSALT-AND-PEPPER NOISE (Transmission Errors):")
    print("  • Root Cause: Bit errors during image transmission/compression")
    print("  • Characteristics: Random black/white spikes")
    best_sp = max(
        restoration_results.get('salt_pepper', {}).items(),
        key=lambda x: x[1]['psnr'],
        default=('N/A', {})
    )
    if best_sp[0] != 'N/A':
        print(f"  • Recommended Filter: {best_sp[0].upper()}")
        print(f"  • Expected PSNR: {best_sp[1]['psnr']:.2f} dB")
    
    print("\n" + "="*70)
    print("PRACTICAL IMPLICATIONS FOR SURVEILLANCE SYSTEMS")
    print("="*70)
    print("\n• Use preprocessing to determine noise type in incoming stream")
    print("• Median filtering preferred for real-time video (handles transmission errors)")
    print("• Gaussian filtering for low-light scene analysis")
    print("• Consider multi-scale filtering for scenes with varying noise levels")
    print("• Trade-off: noise reduction vs. edge preservation")


def main():
    parser = argparse.ArgumentParser(description='Image Restoration - Assignment 2')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to input surveillance image')
    parser.add_argument('--output', '-o', type=str, default='outputs/asmt2',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Determine input image path
    if args.input is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sample_path = os.path.join(base_dir, '../../assets/asmt2/sample_1.jpg')
        
        if os.path.exists(sample_path):
            args.input = sample_path
        else:
            print("Error: No input image found. Please provide --input argument.")
            sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Task 1: Selection and Preprocessing
    original_image = task_1_selection_preprocessing(args.input)
    
    # Task 2: Noise Modeling
    noise_results = task_2_noise_modeling(original_image, args.output)
    
    # Task 3: Restoration
    restoration_results = task_3_restoration(original_image, noise_results, args.output)
    
    # Task 4: Performance Evaluation
    task_4_performance_evaluation(original_image, restoration_results, noise_results)
    
    # Task 5: Analytical Discussion
    task_5_analytical_discussion(original_image, restoration_results, noise_results)
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"\nAll outputs saved to: {os.path.abspath(args.output)}")
    
    plt.close('all')


if __name__ == '__main__':
    main()
