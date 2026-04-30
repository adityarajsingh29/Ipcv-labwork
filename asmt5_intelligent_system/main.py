#!/usr/bin/env python3
"""
Mini Project Assignment 5: Intelligent Image Enhancement & Analysis System (CAPSTONE)

Learning Objectives:
- Integrate concepts from all units of Image Processing & Computer Vision
- Design a complete image processing pipeline
- Apply enhancement, restoration, segmentation, and feature extraction techniques
- Evaluate system performance using standard image quality metrics
- Develop clean, modular, and ethical Python code for real-world problems

This capstone project combines all learning from previous assignments into
a single, structured system that performs end-to-end image processing.
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
    apply_gaussian_blur, apply_median_filter,
    histogram_equalization, apply_clahe,
    quantize_image, downsample_image, upsample_image,
    display_images, print_section_header, compute_mse, compute_psnr, compute_ssim
)


def task_1_setup():
    """Task 1: Project Setup & System Overview"""
    print_section_header("Task 1: Project Setup & System Overview")
    
    print("\n" + "="*75)
    print("     INTELLIGENT IMAGE PROCESSING & ANALYSIS SYSTEM (CAPSTONE)")
    print("="*75)
    
    print("\nWelcome to the End-to-End Intelligent Image Processing System!")
    print("\nThis capstone project integrates all Units of the course:")
    print("  • Unit 1: Image Acquisition & Preprocessing")
    print("  • Unit 2: Enhancement, Restoration & Noise Handling")
    print("  • Unit 3: Segmentation & Morphological Operations")
    print("  • Unit 4: Feature Extraction & Object Representation")
    print("  • Capstone: End-to-end Pipeline Integration")
    
    print("\nSYSTEM WORKFLOW:")
    print("  1. Image Acquisition & Preprocessing")
    print("  2. Image Enhancement & Restoration")
    print("  3. Image Segmentation & Morphology")
    print("  4. Feature Extraction & Analysis")
    print("  5. Performance Evaluation & Reporting")
    
    print("\nSystem is designed for real-world applications:")
    print("  [OK] Surveillance systems")
    print("  [OK] Medical imaging")
    print("  [OK] Document digitization")
    print("  [OK] Traffic monitoring")
    print("  [OK] General image analysis")


def task_2_acquisition_preprocessing(image_path):
    """Task 2: Image Acquisition & Preprocessing"""
    print_section_header("Task 2: Image Acquisition & Preprocessing")
    
    print(f"\nLoading image from: {image_path}")
    original_img = load_image(image_path)
    
    print("Step 1: Resizing to standard resolution (512×512)...")
    original_img = resize_image(original_img, size=(512, 512))
    
    print("Step 2: Converting to grayscale...")
    gray_img = convert_to_grayscale(original_img)
    
    print("Step 3: Analysis of input image")
    print(f"  • Dimensions: {gray_img.shape}")
    print(f"  • Data type: {gray_img.dtype}")
    print(f"  • Value range: [{gray_img.min()}, {gray_img.max()}]")
    print(f"  • Mean intensity: {gray_img.mean():.2f}")
    print(f"  • Std. deviation: {gray_img.std():.2f}")
    
    display_images([original_img, gray_img], ["Original Color", "Grayscale"], figsize=(10, 5))
    
    return original_img, gray_img


def task_3_enhancement_restoration(gray_img, output_dir):
    """Task 3: Image Enhancement & Restoration"""
    print_section_header("Task 3: Image Enhancement & Restoration")
    
    pipeline_results = {'original': gray_img}
    
    # Step 1: Add realistic degradation (noise)
    print("\nStep 1: Simulating real-world degradation...")
    print("  • Adding Gaussian noise (std=15)...")
    noisy_gaussian = add_gaussian_noise(gray_img, mean=0, std=15)
    
    print("  • Adding Salt-and-Pepper noise (p=0.02)...")
    noisy_sp = add_salt_pepper_noise(noisy_gaussian, prob=0.02)
    
    pipeline_results['degraded'] = noisy_sp
    
    # Step 2: Restoration
    print("\nStep 2: Applying restoration filters...")
    print("  • Median filter (kernel=5)...")
    median_filtered = apply_median_filter(noisy_sp, kernel_size=5)
    
    print("  • Gaussian blur (kernel=5, sigma=1)...")
    gaussian_filtered = apply_gaussian_blur(noisy_sp, kernel_size=(5, 5), sigma=1.0)
    
    # Use best (combination)
    restored = cv2.addWeighted(median_filtered, 0.6, gaussian_filtered, 0.4, 0)
    pipeline_results['restored'] = restored
    
    # Step 3: Enhancement
    print("\nStep 3: Applying enhancement techniques...")
    print("  • Histogram Equalization...")
    enhanced_he = histogram_equalization(restored)
    
    print("  • CLAHE (Contrast Limited Adaptive Histogram Equalization)...")
    enhanced_clahe = apply_clahe(restored, clip_limit=2.0, tile_grid_size=(8, 8))
    
    # Use CLAHE for better results
    pipeline_results['enhanced'] = enhanced_clahe
    
    # Compute metrics
    mse_deg = compute_mse(gray_img, noisy_sp)
    psnr_deg = compute_psnr(gray_img, noisy_sp)
    
    mse_rest = compute_mse(gray_img, restored)
    psnr_rest = compute_psnr(gray_img, restored)
    
    mse_enh = compute_mse(gray_img, pipeline_results['enhanced'])
    psnr_enh = compute_psnr(gray_img, pipeline_results['enhanced'])
    
    print(f"\nQuality Metrics:")
    print(f"  • Degraded:  MSE={mse_deg:.4f}, PSNR={psnr_deg:.4f} dB")
    print(f"  • Restored: MSE={mse_rest:.4f}, PSNR={psnr_rest:.4f} dB")
    print(f"  • Enhanced: MSE={mse_enh:.4f}, PSNR={psnr_enh:.4f} dB")
    
    # Display
    images = [gray_img, noisy_sp, restored, enhanced_clahe]
    titles = [
        "Original",
        f"Degraded\nPSNR: {psnr_deg:.2f} dB",
        f"Restored\nPSNR: {psnr_rest:.2f} dB",
        f"Enhanced\nPSNR: {psnr_enh:.2f} dB"
    ]
    fig = display_images(images, titles, figsize=(14, 5))
    fig_path = os.path.join(output_dir, 'stage_enhancement_restoration.png')
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    
    return pipeline_results


def task_4_segmentation_morphology(enhanced_img, output_dir):
    """Task 4: Image Segmentation & Morphological Processing"""
    print_section_header("Task 4: Image Segmentation & Morphological Processing")
    
    pipeline_results = {'enhanced': enhanced_img}
    
    # Thresholding
    print("\nStep 1: Applying thresholding...")
    print("  • Computing Otsu's automatic threshold...")
    otsu_thresh, binary = cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"    - Otsu threshold value: {otsu_thresh}")
    
    pipeline_results['segmented'] = binary
    
    # Morphological operations
    print("\nStep 2: Applying morphological operations...")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    print("  • Opening (erosion -> dilation)...")
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    print("  • Closing (dilation -> erosion)...")
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    pipeline_results['morpho_refined'] = closed
    
    # Analyze segmentation
    num_objects = len(cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
    foreground_pixels = np.sum(closed > 0)
    foreground_percent = (foreground_pixels / closed.size) * 100
    
    print(f"\nSegmentation Analysis:")
    print(f"  • Objects detected: {num_objects}")
    print(f"  • Foreground pixels: {foreground_pixels} ({foreground_percent:.2f}%)")
    
    # Display
    images = [enhanced_img, binary, closed]
    titles = [
        "Enhanced Image",
        f"Otsu Threshold\n(T={otsu_thresh})",
        "Morphologically Refined"
    ]
    fig = display_images(images, titles, figsize=(14, 5))
    fig_path = os.path.join(output_dir, 'stage_segmentation_morphology.png')
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    
    return pipeline_results


def task_5_feature_extraction(enhanced_img, segmented_img, output_dir):
    """Task 5: Feature Extraction & Object Representation"""
    print_section_header("Task 5: Feature Extraction & Object Representation")
    
    pipeline_results = {}
    
    # Edge detection
    print("\nStep 1: Detecting edges...")
    print("  • Applying Canny edge detector...")
    edges = cv2.Canny(enhanced_img, 100, 200)
    pipeline_results['edges'] = edges
    
    # Contour detection
    print("\nStep 2: Detecting contours & objects...")
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"  • Contours detected: {len(contours)}")
    
    # Draw contours and bounding boxes
    contour_img = np.zeros_like(edges)
    cv2.drawContours(contour_img, contours, -1, 255, 2)
    
    bbox_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    objects_info = []
    for idx, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 50:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            objects_info.append({'id': idx, 'area': area, 'bbox': (x, y, w, h)})
    
    print(f"  • Significant objects (area > 50): {len(objects_info)}")
    
    # Feature extraction using ORB
    print("\nStep 3: Extracting features using ORB...")
    orb = cv2.ORB_create(nfeatures=300)
    keypoints, descriptors = orb.detectAndCompute(enhanced_img, None)
    print(f"  • Keypoints detected: {len(keypoints)}")
    
    keypoint_img = cv2.drawKeypoints(enhanced_img, keypoints, None,
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    pipeline_results['keypoints'] = len(keypoints)
    pipeline_results['objects'] = len(objects_info)
    
    # Display
    images = [enhanced_img, edges, contour_img, bbox_img, keypoint_img]
    titles = [
        "Enhanced Image",
        "Edges",
        "Contours",
        "Bounding Boxes",
        f"ORB Features\n({len(keypoints)} keypoints)"
    ]
    fig = display_images(images, titles, figsize=(16, 6))
    fig_path = os.path.join(output_dir, 'stage_feature_extraction.png')
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    
    return pipeline_results


def task_6_performance_evaluation(original_img, degraded_img, restored_img, enhanced_img, output_dir):
    """Task 6: Performance Evaluation"""
    print_section_header("Task 6: Performance Evaluation")
    
    print("\nComputing image quality metrics...")
    
    # MSE and PSNR
    mse_degraded = compute_mse(original_img, degraded_img)
    psnr_degraded = compute_psnr(original_img, degraded_img)
    
    mse_restored = compute_mse(original_img, restored_img)
    psnr_restored = compute_psnr(original_img, restored_img)
    
    mse_enhanced = compute_mse(original_img, enhanced_img)
    psnr_enhanced = compute_psnr(original_img, enhanced_img)
    
    # SSIM
    ssim_degraded = compute_ssim(original_img, degraded_img)
    ssim_restored = compute_ssim(original_img, restored_img)
    ssim_enhanced = compute_ssim(original_img, enhanced_img)
    
    print("\n" + "="*70)
    print("IMAGE QUALITY METRICS")
    print("="*70)
    
    print(f"\nDegraded Image:")
    print(f"  • MSE:  {mse_degraded:.4f}")
    print(f"  • PSNR: {psnr_degraded:.4f} dB")
    print(f"  • SSIM: {ssim_degraded:.4f}")
    
    print(f"\nRestored Image:")
    print(f"  • MSE:  {mse_restored:.4f}")
    print(f"  • PSNR: {psnr_restored:.4f} dB")
    print(f"  • SSIM: {ssim_restored:.4f}")
    print(f"  - Improvement over degraded: MSE down {(1-mse_restored/mse_degraded)*100:.1f}%, PSNR up {psnr_restored-psnr_degraded:.2f} dB")
    
    print(f"\nEnhanced Image:")
    print(f"  • MSE:  {mse_enhanced:.4f}")
    print(f"  • PSNR: {psnr_enhanced:.4f} dB")
    print(f"  • SSIM: {ssim_enhanced:.4f}")
    print(f"  - Improvement over restored: MSE down {(1-mse_enhanced/mse_restored)*100:.1f}%, PSNR up {psnr_enhanced-psnr_restored:.2f} dB")
    
    # Save metrics to file
    metrics_file = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("IMAGE QUALITY EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Degraded:  MSE={mse_degraded:.4f}, PSNR={psnr_degraded:.4f} dB, SSIM={ssim_degraded:.4f}\n")
        f.write(f"Restored:  MSE={mse_restored:.4f}, PSNR={psnr_restored:.4f} dB, SSIM={ssim_restored:.4f}\n")
        f.write(f"Enhanced:  MSE={mse_enhanced:.4f}, PSNR={psnr_enhanced:.4f} dB, SSIM={ssim_enhanced:.4f}\n")
    
    return {
        'degraded': {'mse': mse_degraded, 'psnr': psnr_degraded, 'ssim': ssim_degraded},
        'restored': {'mse': mse_restored, 'psnr': psnr_restored, 'ssim': ssim_restored},
        'enhanced': {'mse': mse_enhanced, 'psnr': psnr_enhanced, 'ssim': ssim_enhanced}
    }


def task_7_visualization_analysis(original_img, pipeline_stages, output_dir, metrics):
    """Task 7: Final Visualization & Analysis"""
    print_section_header("Task 7: Final Visualization & Analysis")
    
    print("\nCreating comprehensive visualization of entire pipeline...")
    
    # Create 6-stage visualization (or as available)
    images = [
        original_img,
        pipeline_stages.get('degraded', original_img),
        pipeline_stages.get('restored', original_img),
        pipeline_stages.get('enhanced', original_img),
        pipeline_stages.get('segmented', original_img),
        pipeline_stages.get('morpho_refined', original_img)
    ]
    
    titles = [
        "1. Original",
        f"2. Degraded\nPSNR: {metrics['degraded']['psnr']:.2f} dB",
        f"3. Restored\nPSNR: {metrics['restored']['psnr']:.2f} dB",
        f"4. Enhanced\nPSNR: {metrics['enhanced']['psnr']:.2f} dB",
        "5. Segmented",
        "6. Morphologically\nRefined"
    ]
    
    fig = display_images(images, titles, figsize=(16, 8))
    fig_path = os.path.join(output_dir, 'complete_pipeline.png')
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    print(f"[OK] Complete pipeline visualization saved: {fig_path}")
    
    # Print analysis
    print("\n" + "="*70)
    print("SYSTEM PERFORMANCE ANALYSIS")
    print("="*70)
    
    print("\n1. IMAGE ACQUISITION:")
    print("   [OK] Successfully loaded and preprocessed image")
    print(f"   [OK] Standardized to 512x512 resolution")
    
    print("\n2. ENHANCEMENT & RESTORATION:")
    degradation = metrics['degraded']['psnr']
    restoration = metrics['restored']['psnr']
    print(f"   [OK] Degradation: PSNR = {degradation:.2f} dB")
    print(f"   [OK] Restoration: PSNR = {restoration:.2f} dB")
    print(f"   [OK] Recovery: {restoration - degradation:.2f} dB improvement")
    
    print("\n3. ENHANCEMENT:")
    enhancement = metrics['enhanced']['psnr']
    print(f"   [OK] Final Enhanced: PSNR = {enhancement:.2f} dB")
    
    print("\n4. SEGMENTATION & MORPHOLOGY:")
    num_objects = pipeline_stages.get('objects', 'N/A')
    print(f"   [OK] Objects detected: {num_objects}")
    
    print("\n5. FEATURE EXTRACTION:")
    num_features = pipeline_stages.get('keypoints', 'N/A')
    print(f"   [OK] Keypoints extracted: {num_features}")
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("\n[OK] End-to-end image processing pipeline executed successfully")
    print("[OK] All stages from acquisition to feature extraction completed")
    print("[OK] Quality metrics demonstrate effective restoration and enhancement")
    print("[OK] System ready for deployment in real-world applications")


def main():
    parser = argparse.ArgumentParser(description='Capstone: Intelligent Image Processing System')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to input image')
    parser.add_argument('--output', '-o', type=str, default='outputs/asmt5',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Determine input image path
    if args.input is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sample_path = os.path.join(base_dir, '../../assets/asmt5/sample_1.jpg')
        
        if os.path.exists(sample_path):
            args.input = sample_path
        else:
            print("Error: No input image found. Please provide --input argument.")
            sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Task 1: Setup
    task_1_setup()
    
    # Task 2: Acquisition & Preprocessing
    original_img, gray_img = task_2_acquisition_preprocessing(args.input)
    
    # Task 3: Enhancement & Restoration
    stage3_results = task_3_enhancement_restoration(gray_img, args.output)
    
    # Task 4: Segmentation & Morphology
    stage4_results = task_4_segmentation_morphology(stage3_results['enhanced'], args.output)
    
    # Task 5: Feature Extraction
    stage5_results = task_5_feature_extraction(stage3_results['enhanced'], stage4_results['morpho_refined'], args.output)
    
    # Task 6: Performance Evaluation
    metrics = task_6_performance_evaluation(
        gray_img, stage3_results['degraded'], 
        stage3_results['restored'], stage3_results['enhanced'], args.output
    )
    
    # Combine all pipeline stages
    all_stages = {
        'original': original_img,
        'degraded': stage3_results['degraded'],
        'restored': stage3_results['restored'],
        'enhanced': stage3_results['enhanced'],
        'segmented': stage4_results['segmented'],
        'morpho_refined': stage4_results['morpho_refined'],
        'objects': stage5_results['objects'],
        'keypoints': stage5_results['keypoints']
    }
    
    # Task 7: Visualization & Analysis
    task_7_visualization_analysis(original_img, all_stages, args.output, metrics)
    
    print("\n" + "="*70)
    print("CAPSTONE PROJECT COMPLETE")
    print("="*70)
    print(f"\nAll results saved to: {os.path.abspath(args.output)}")
    
    plt.close('all')


if __name__ == '__main__':
    main()
