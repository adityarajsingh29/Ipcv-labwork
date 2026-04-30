#!/usr/bin/env python3
"""
Mini Project Assignment 4: Feature-Based Traffic Monitoring System

Learning Objectives:
- Detect edges and object boundaries
- Represent objects using contours
- Extract key features using classical CV techniques
- Understand how features support higher-level vision tasks

This project simulates a feature-based traffic analysis system.
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


def task_1_edge_detection(gray_image, output_dir):
    """Task 1: Edge Detection"""
    print_section_header("Task 1: Edge Detection")
    
    edge_results = {}
    
    # Sobel Edge Detection
    print("\nApplying Sobel Edge Detector...")
    sobel_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    sobel = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel = np.clip(sobel, 0, 255).astype(np.uint8)
    
    edge_results['sobel'] = sobel
    save_image(sobel, os.path.join(output_dir, 'edges_sobel.jpg'))
    print("  [OK] Sobel edge detection complete")
    print(f"    - Sobel response range: [{sobel.min()}, {sobel.max()}]")
    
    # Canny Edge Detector
    print("\nApplying Canny Edge Detector...")
    canny = cv2.Canny(gray_image, 100, 200)
    
    edge_results['canny'] = canny
    save_image(canny, os.path.join(output_dir, 'edges_canny.jpg'))
    print("  [OK] Canny edge detection complete")
    print(f"    - Threshold values: low=100, high=200")
    
    # Display edge detection results
    images = [gray_image, sobel, canny]
    titles = [
        "Original Image",
        "Sobel Edges",
        "Canny Edges"
    ]
    fig = display_images(images, titles, figsize=(14, 5))
    fig_path = os.path.join(output_dir, 'edge_detection_comparison.png')
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    print(f"\nEdge detection comparison saved to: {fig_path}")
    
    return edge_results


def task_2_object_representation(edge_results, output_dir):
    """Task 2: Object Representation (Contours & Bounding Boxes)"""
    print_section_header("Task 2: Object Representation")
    
    object_results = {}
    
    for edge_type, edges in edge_results.items():
        print(f"\nDetecting contours from {edge_type.upper()} edges...")
        
        # Detect contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  • Number of contours found: {len(contours)}")
        
        # Create visualization
        contour_image = np.zeros_like(edges)
        cv2.drawContours(contour_image, contours, -1, 255, 2)
        
        # Compute object properties
        objects = []
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Only consider significant contours
            if area > 50:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(contour)
                
                objects.append({
                    'id': idx,
                    'area': area,
                    'perimeter': perimeter,
                    'bbox': (x, y, w, h),
                    'contour': contour
                })
        
        print(f"  • Significant objects (area > 50): {len(objects)}")
        
        # Create bounding box image
        bbox_image = np.copy(edges)
        bbox_image = cv2.cvtColor(bbox_image, cv2.COLOR_GRAY2BGR)
        
        for obj in objects:
            x, y, w, h = obj['bbox']
            cv2.rectangle(bbox_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(bbox_image, f"A:{obj['area']:.0f}", (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        object_results[edge_type] = {
            'contours': contours,
            'contour_image': contour_image,
            'bbox_image': bbox_image,
            'objects': objects
        }
        
        save_image(contour_image, os.path.join(output_dir, f'contours_{edge_type}.jpg'))
        save_image(bbox_image, os.path.join(output_dir, f'bounding_boxes_{edge_type}.jpg'))
        
        # Print measurements for first few objects
        if objects:
            print(f"\n  First {min(3, len(objects))} objects detected:")
            for obj in objects[:3]:
                print(f"    • Object {obj['id']}: Area={obj['area']:.2f}, Perimeter={obj['perimeter']:.2f}")
    
    # Display comparison
    images = [
        object_results['sobel']['contour_image'],
        object_results['canny']['contour_image'],
        object_results['sobel']['bbox_image'],
        object_results['canny']['bbox_image']
    ]
    titles = [
        "Sobel Contours",
        "Canny Contours",
        "Sobel Bounding Boxes",
        "Canny Bounding Boxes"
    ]
    fig = display_images(images, titles, figsize=(14, 10))
    fig_path = os.path.join(output_dir, 'object_representation.png')
    plt.savefig(fig_path, dpi=100, bbox_inches='tight')
    print(f"\nObject representation figure saved to: {fig_path}")
    
    return object_results


def task_3_feature_extraction(gray_image, output_dir):
    """Task 3: Feature Extraction using ORB"""
    print_section_header("Task 3: Feature Extraction")
    
    print("\nInitializing ORB (Oriented FAST and Rotated BRIEF) detector...")
    print("  - ORB: Efficient alternative to SIFT/SURF")
    print("  - Uses FAST corner detection + BRIEF descriptors")
    
    # Create ORB detector
    orb = cv2.ORB_create(nfeatures=500)
    
    # Detect keypoints and descriptors
    print("\nDetecting keypoints and computing descriptors...")
    keypoints, descriptors = orb.detectAndCompute(gray_image, None)
    
    print(f"  [OK] Keypoints detected: {len(keypoints)}")
    if descriptors is not None:
        print(f"  [OK] Descriptors computed: {descriptors.shape}")
    
    # Visualize keypoints
    keypoint_image = cv2.drawKeypoints(gray_image, keypoints, None,
                                       flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    save_image(keypoint_image, os.path.join(output_dir, 'features_orb_keypoints.jpg'))
    print(f"  [OK] Keypoint visualization saved")
    
    # Print keypoint statistics
    if keypoints:
        angles = [kp.angle for kp in keypoints]
        sizes = [kp.size for kp in keypoints]
        
        print(f"\n  Keypoint Statistics:")
        print(f"    - Angle range: [{min(angles):.2f}, {max(angles):.2f}] degrees")
        print(f"    - Size range: [{min(sizes):.2f}, {max(sizes):.2f}]")
    
    feature_results = {
        'keypoints': keypoints,
        'descriptors': descriptors,
        'keypoint_image': keypoint_image,
        'num_features': len(keypoints)
    }
    
    return feature_results


def task_4_comparative_analysis(edge_results, object_results, feature_results):
    """Task 4: Comparative Analysis"""
    print_section_header("Task 4: Comparative Analysis")
    
    print("\n" + "="*70)
    print("EDGE DETECTOR COMPARISON")
    print("="*70)
    
    print("\nSOBEL EDGE DETECTOR:")
    print("  • Method: Gradient-based (Sobel operator)")
    print("  • Advantages:")
    print("    - Simple and computationally efficient")
    print("    - Good for detecting strong edges")
    print("    - Directional information (x, y gradients)")
    print("  • Disadvantages:")
    print("    - Sensitive to noise")
    print("    - Non-maximum suppression not applied")
    
    sobel_contours = len(object_results['sobel']['contours'])
    print(f"  • Contours detected: {sobel_contours}")
    
    print("\nCANNY EDGE DETECTOR:")
    print("  • Method: Multi-stage (gradient -> non-max suppression -> hysteresis)")
    print("  • Advantages:")
    print("    - More sophisticated edge detection")
    print("    - Non-maximum suppression reduces thick edges")
    print("    - Hysteresis thresholding for edge linking")
    print("    - Better noise robustness")
    print("  • Disadvantages:")
    print("    - More computationally expensive")
    print("    - Threshold parameter tuning required")
    
    canny_contours = len(object_results['canny']['contours'])
    print(f"  • Contours detected: {canny_contours}")
    
    print("\nRECOMMENDATION FOR TRAFFIC MONITORING:")
    if canny_contours > sobel_contours * 0.8 and canny_contours < sobel_contours * 1.2:
        print("  -> Canny edge detector recommended for better accuracy")
    else:
        print("  -> Choose based on specific scene characteristics and requirements")
    
    print("\n" + "="*70)
    print("FEATURE EXTRACTION & ITS ROLE IN TRAFFIC MONITORING")
    print("="*70)
    
    print("\nORB FEATURE EXTRACTOR:")
    print(f"  • Features detected: {feature_results['num_features']}")
    print("  • Applications in traffic monitoring:")
    print("    1. Vehicle tracking across frames")
    print("    2. Matching vehicles at different viewing angles")
    print("    3. Robust to scale and rotation changes")
    print("    4. Feature-based image matching for re-identification")
    
    print("\nFEATURE MATCHING USE CASES:")
    print("  • Multi-camera tracking: Match same vehicle across cameras")
    print("  • Temporal tracking: Link vehicle detections across video frames")
    print("  • Vehicle re-identification: Recognize same vehicle after occlusion")
    print("  • License plate correlation: Verify vehicle features")
    
    print("\n" + "="*70)
    print("INTEGRATED TRAFFIC SYSTEM WORKFLOW")
    print("="*70)
    print("\n1. Edge Detection -> Object boundaries identified")
    print("2. Contour Detection -> Vehicle shapes extracted")
    print("3. Bounding Boxes -> Spatial localization")
    print("4. Feature Extraction -> Distinctive characteristics captured")
    print("5. Feature Matching -> Cross-frame/cross-camera tracking")
    print("6. Classification -> Vehicle type, speed, direction determined")


def main():
    parser = argparse.ArgumentParser(description='Traffic Monitoring - Assignment 4')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Path to input traffic image')
    parser.add_argument('--output', '-o', type=str, default='outputs/asmt4',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Determine input image path
    if args.input is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sample_path = os.path.join(base_dir, '../../assets/asmt4/sample_1.jpg')
        
        if os.path.exists(sample_path):
            args.input = sample_path
        else:
            print("Error: No input image found. Please provide --input argument.")
            sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load and preprocess
    print_section_header("Image Loading & Preprocessing")
    print(f"\nLoading traffic image from: {args.input}")
    image = load_image(args.input)
    gray_image = convert_to_grayscale(image)
    gray_image = resize_image(gray_image, size=(512, 512))
    print(f"Image preprocessed: {gray_image.shape}")
    
    # Task 1: Edge Detection
    edge_results = task_1_edge_detection(gray_image, args.output)
    
    # Task 2: Object Representation
    object_results = task_2_object_representation(edge_results, args.output)
    
    # Task 3: Feature Extraction
    feature_results = task_3_feature_extraction(gray_image, args.output)
    
    # Task 4: Comparative Analysis
    task_4_comparative_analysis(edge_results, object_results, feature_results)
    
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"\nAll outputs saved to: {os.path.abspath(args.output)}")
    
    plt.close('all')


if __name__ == '__main__':
    main()
