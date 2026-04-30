#!/usr/bin/env python3
"""
Mini Project Assignment 3: Medical Image Compression & Segmentation System


Learning Objectives:
- Understand image data redundancy
- Implement lossless compression techniques
- Perform image segmentation using thresholding
- Improve segmentation using morphological operations

This project focuses on compressing medical images and segmenting meaningful regions.
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
    display_images, print_section_header
)


def run_length_encode(data):
    """
    Run Length Encoding (RLE) for lossless compression.
    
    Args:
        data: 1D or flattened array of data
    
    Returns:
        Tuple of (encoded_values, run_lengths)
    """
    data = data.flatten()
    encoded_values = []
    run_lengths = []
    
    if len(data) == 0:
        return np.array(encoded_values), np.array(run_lengths)
    
    current_value = data[0]
    current_count = 1
    
    for i in range(1, len(data)):
        if data[i] == current_value and current_count < 255:
            current_count += 1
        else:
            encoded_values.append(current_value)
            run_lengths.append(current_count)
            current_value = data[i]
            current_count = 1
    
    # Don't forget the last run
    encoded_values.append(current_value)
    run_lengths.append(current_count)
    
    return np.array(encoded_values, dtype=np.uint8), np.array(run_lengths, dtype=np.uint8)


def run_length_decode(encoded_values, run_lengths):
    """
    Run Length Decoding to reconstruct original data.
    
    Args:
        encoded_values: Array of encoded values
        run_lengths: Array of run lengths
    
    Returns:
        Reconstructed 1D array
    """
    decoded = []
    for value, count in zip(encoded_values, run_lengths):
        decoded.extend([value] * count)
    return np.array(decoded, dtype=np.uint8)


def calculate_compression_metrics(original_size, compressed_size):
    """Calculate compression ratio and storage savings."""
    ratio = original_size / compressed_size if compressed_size > 0 else 0
    savings_percent = ((original_size - compressed_size) / original_size * 100) if original_size > 0 else 0
    return ratio, savings_percent


def task_1_compression(gray_image, output_dir):
    """Task 1: Image Compression using RLE"""
    print_section_header("Task 1: Image Compression")
    
    print("\nApplying Run Length Encoding (RLE) compression...")
    
    # Get original size
    original_size = gray_image.nbytes
    print(f"\nOriginal image size: {original_size} bytes ({original_size / 1024:.2f} KB)")
    
    # Apply RLE
    encoded_values, run_lengths = run_length_encode(gray_image)
    
    # Calculate compressed size
    # Overhead: storing the shape and data
    compressed_size = encoded_values.nbytes + run_lengths.nbytes + 8  # +8 for shape info
    
    print(f"Encoded values size: {encoded_values.nbytes} bytes")
    print(f"Run lengths size: {run_lengths.nbytes} bytes")
    print(f"Total compressed size: {compressed_size} bytes ({compressed_size / 1024:.2f} KB)")
    
    # Calculate metrics
    ratio, savings = calculate_compression_metrics(original_size, compressed_size)
    
    print(f"\nCompression Ratio: {ratio:.4f}x")
    print(f"Storage Savings: {savings:.2f}%")
    
    # Decompress to verify
    print("\nVerifying decompression...")
    decompressed = run_length_decode(encoded_values, run_lengths)
    decompressed = decompressed[:gray_image.size].reshape(gray_image.shape)
    
    # Check if lossless
    error = np.sum(gray_image != decompressed)
    print(f"Reconstruction error pixels: {error} / {gray_image.size}")
    
    if error == 0:
        print("[OK] Lossless compression verified - perfect reconstruction")
    else:
        print(f"[WARN] Warning: {error} pixels differ after decompression")
    
    compression_info = {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'ratio': ratio,
        'savings': savings,
        'encoded_values': encoded_values,
        'run_lengths': run_lengths
    }
    
    return compression_info


def task_2_segmentation(gray_image, output_dir):
    """Task 2: Image Segmentation"""
    print_section_header("Task 2: Image Segmentation")
    
    segmentation_results = {}
    
    # Global Thresholding
    print("\nApplying Global Thresholding (threshold=127)...")
    _, global_thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    segmentation_results['global'] = global_thresh
    save_image(global_thresh, os.path.join(output_dir, 'segmented_global_threshold.jpg'))
    print("  [OK] Global threshold segmentation complete")
    
    # Otsu's Thresholding
    print("\nApplying Otsu's Adaptive Thresholding...")
    otsu_thresh, otsu_binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"  - Otsu threshold value: {otsu_thresh}")
    segmentation_results['otsu'] = otsu_binary
    save_image(otsu_binary, os.path.join(output_dir, 'segmented_otsu_threshold.jpg'))
    print("  [OK] Otsu thresholding complete")
    
    # Display segmentation results
    images = [gray_image, global_thresh, otsu_binary]
    titles = [
        "Original",
        f"Global Threshold (T=127)",
        f"Otsu Threshold (T={otsu_thresh})"
    ]
    fig = display_images(images, titles, figsize=(14, 5))
    fig_path = os.path.join(output_dir, 'segmentation_comparison.png')
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    print(f"\nSegmentation comparison figure saved to: {fig_path}")
    
    return segmentation_results


def task_3_morphological_processing(segmentation_results, output_dir):
    """Task 3: Morphological Processing"""
    print_section_header("Task 3: Morphological Processing")
    
    morphological_results = {}
    
    # Create morphological kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    for seg_method, binary_image in segmentation_results.items():
        print(f"\nApplying morphological operations to {seg_method} segmentation...")
        
        # Erosion
        eroded = cv2.erode(binary_image, kernel, iterations=1)
        print(f"  - Erosion applied (1 iteration)")
        
        # Dilation
        dilated = cv2.dilate(binary_image, kernel, iterations=1)
        print(f"  - Dilation applied (1 iteration)")
        
        # Opening (erosion followed by dilation) - removes small noise
        opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
        print(f"  - Opening applied (removes small noise)")
        
        # Closing (dilation followed by erosion) - fills small holes
        closed = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        print(f"  - Closing applied (fills holes)")
        
        morphological_results[seg_method] = {
            'eroded': eroded,
            'dilated': dilated,
            'opened': opened,
            'closed': closed
        }
        
        # Save results
        save_image(eroded, os.path.join(output_dir, f'morpho_{seg_method}_eroded.jpg'))
        save_image(dilated, os.path.join(output_dir, f'morpho_{seg_method}_dilated.jpg'))
        save_image(opened, os.path.join(output_dir, f'morpho_{seg_method}_opened.jpg'))
        save_image(closed, os.path.join(output_dir, f'morpho_{seg_method}_closed.jpg'))
        
        # Display morphological operations
        images = [binary_image, eroded, dilated, opened, closed]
        titles = [
            "Original Segmentation",
            "Eroded",
            "Dilated",
            "Opened",
            "Closed"
        ]
        fig = display_images(images, titles, figsize=(16, 5))
        fig_path = os.path.join(output_dir, f'morpho_{seg_method}_operations.png')
        plt.savefig(fig_path, dpi=100, bbox_inches='tight')
        print(f"  [OK] Morphological operations figure saved")
    
    return morphological_results


def task_4_analysis(gray_image, compression_info, segmentation_results, morphological_results):
    """Task 4: Analysis and Interpretation"""
    print_section_header("Task 4: Analysis and Interpretation")
    
    print("\n" + "="*70)
    print("COMPRESSION ANALYSIS")
    print("="*70)
    print(f"\nOriginal Size: {compression_info['original_size']} bytes ({compression_info['original_size']/1024:.2f} KB)")
    print(f"Compressed Size: {compression_info['compressed_size']} bytes ({compression_info['compressed_size']/1024:.2f} KB)")
    print(f"Compression Ratio: {compression_info['ratio']:.4f}x")
    print(f"Storage Savings: {compression_info['savings']:.2f}%")
    
    if compression_info['ratio'] < 1.2:
        print("\nObservation: Limited compression achieved")
        print("  • Reason: Natural images have high entropy (less redundancy)")
        print("  • RLE effective for: Binary images, text, simple graphics")
        print("  • Recommendation: Use JPEG/PNG for better compression of medical images")
    else:
        print(f"\nObservation: Good compression achieved ({compression_info['ratio']:.2f}x)")
        print("  • Medical images may have large uniform regions")
    
    print("\n" + "="*70)
    print("SEGMENTATION ANALYSIS")
    print("="*70)
    
    for seg_method, binary_image in segmentation_results.items():
        # Count foreground pixels
        foreground_pixels = np.sum(binary_image > 0)
        foreground_percent = (foreground_pixels / binary_image.size) * 100
        
        print(f"\n{seg_method.upper()} Thresholding:")
        print(f"  • Foreground pixels: {foreground_pixels} ({foreground_percent:.2f}%)")
        print(f"  • Background pixels: {binary_image.size - foreground_pixels} ({100-foreground_percent:.2f}%)")
    
    print("\n" + "="*70)
    print("CLINICAL RELEVANCE")
    print("="*70)
    print("\n• Segmentation identifies regions of interest (tumors, bones, organs)")
    print("• Global thresholding: Simple, fast, suitable for clear boundaries")
    print("• Otsu thresholding: Adaptive, better for varying image conditions")
    print("• Morphological operations: Refine boundaries, remove artifacts")
    print("\nMorphological Impact:")
    print("  • Erosion: Removes thin features, shrinks objects")
    print("  • Dilation: Fills holes, enlarges objects")
    print("  • Opening: Removes small noise objects")
    print("  • Closing: Fills small holes within objects")


def main():
    parser = argparse.ArgumentParser(description='Medical Image System - Assignment 3')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to input medical image')
    parser.add_argument('--output', '-o', type=str, default='outputs/asmt3',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Determine input image path
    if args.input is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sample_path = os.path.join(base_dir, '../../assets/asmt3/sample_1.jpg')
        
        if os.path.exists(sample_path):
            args.input = sample_path
        else:
            print("Error: No input image found. Please provide --input argument.")
            sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load and preprocess
    print_section_header("Image Loading & Preprocessing")
    print(f"\nLoading medical image from: {args.input}")
    gray_image = load_image(args.input, grayscale=True)
    gray_image = resize_image(gray_image, size=(512, 512))
    print(f"Image preprocessed: {gray_image.shape}")
    
    # Task 1: Compression
    compression_info = task_1_compression(gray_image, args.output)
    
    # Task 2: Segmentation
    segmentation_results = task_2_segmentation(gray_image, args.output)
    
    # Task 3: Morphological Processing
    morphological_results = task_3_morphological_processing(segmentation_results, args.output)
    
    # Task 4: Analysis
    task_4_analysis(gray_image, compression_info, segmentation_results, morphological_results)
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"\nAll outputs saved to: {os.path.abspath(args.output)}")
    
    plt.close('all')


if __name__ == '__main__':
    main()
