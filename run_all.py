#!/usr/bin/env python3
"""
Run all five assignments on sample images and generate a summary report.

This script tests each assignment with different sample images and collects
metrics into a comprehensive report.
"""

import os
import sys
import subprocess
from pathlib import Path


def run_assignment(script_path, sample_image, output_dir):
    """Run an assignment script and return success status."""
    try:
        cmd = [
            sys.executable, script_path,
            '--input', sample_image,
            '--output', output_dir
        ]
        print(f"  Running: {' '.join(cmd[:3])}... ", end='')
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("[OK] SUCCESS")
            return True
        else:
            print("[FAIL] FAILED")
            print(f"    Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("[FAIL] TIMEOUT")
        return False
    except Exception as e:
        print(f"[FAIL] ERROR: {e}")
        return False


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("\n" + "="*75)
    print("RUN ALL ASSIGNMENTS - TEST HARNESS")
    print("="*75 + "\n")
    
    assignments = [
        {
            'name': 'Assignment 1: Document Scanner',
            'script': os.path.join(base_dir, 'asmt1_scanner/scanner.py'),
            'samples': [f'assets/asmt1/sample_{i}.jpg' for i in range(1, 4)],
            'output_prefix': 'outputs/asmt1_run'
        },
        {
            'name': 'Assignment 2: Image Restoration',
            'script': os.path.join(base_dir, 'asmt2_restoration/restoration.py'),
            'samples': [f'assets/asmt2/sample_{i}.jpg' for i in range(1, 4)],
            'output_prefix': 'outputs/asmt2_run'
        },
        {
            'name': 'Assignment 3: Medical Imaging',
            'script': os.path.join(base_dir, 'asmt3_medical/medical_image_system.py'),
            'samples': [f'assets/asmt3/sample_{i}.jpg' for i in range(1, 4)],
            'output_prefix': 'outputs/asmt3_run'
        },
        {
            'name': 'Assignment 4: Traffic Monitoring',
            'script': os.path.join(base_dir, 'asmt4_traffic/traffic_monitoring.py'),
            'samples': [f'assets/asmt4/sample_{i}.jpg' for i in range(1, 4)],
            'output_prefix': 'outputs/asmt4_run'
        },
        {
            'name': 'Assignment 5: Capstone',
            'script': os.path.join(base_dir, 'asmt5_intelligent_system/main.py'),
            'samples': [f'assets/asmt5/sample_{i}.jpg' for i in range(1, 4)],
            'output_prefix': 'outputs/asmt5_run'
        }
    ]
    
    results = {}
    
    for assignment in assignments:
        print(f"\n{'='*75}")
        print(f"{assignment['name']}")
        print(f"{'='*75}")
        
        script_path = assignment['script']
        
        if not os.path.exists(script_path):
            print(f"  [FAIL] Script not found: {script_path}")
            results[assignment['name']] = False
            continue
        
        success_count = 0
        total_count = 0
        
        for idx, sample in enumerate(assignment['samples'], 1):
            sample_path = os.path.join(base_dir, sample)
            
            if not os.path.exists(sample_path):
                print(f"  Sample {idx}: [FAIL] Image not found: {sample_path}")
                continue
            
            output_dir = f"{assignment['output_prefix']}_{idx}"
            output_path = os.path.join(base_dir, output_dir)
            
            print(f"  Sample {idx}:", end=' ')
            if run_assignment(script_path, sample_path, output_path):
                success_count += 1
            
            total_count += 1
        
        assignment_success = (success_count == total_count) and total_count > 0
        results[assignment['name']] = assignment_success
        
        if assignment_success:
            print(f"\n  [OK] All {total_count} samples completed successfully")
        else:
            print(f"\n  [FAIL] {success_count}/{total_count} samples completed")
    
    # Print summary
    print(f"\n{'='*75}")
    print("TEST SUMMARY")
    print(f"{'='*75}\n")
    
    for name, success in results.items():
        status = "[OK] PASS" if success else "[FAIL] FAIL"
        print(f"{status} - {name}")
    
    total_pass = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n{'-'*75}")
    print(f"Total: {total_pass}/{total} assignments passed")
    print(f"\nAll outputs saved to: {os.path.join(base_dir, 'outputs/')}")
    
    if total_pass == total:
        print("\n[OK] ALL TESTS PASSED!")
        return 0
    else:
        print("\n[FAIL] Some tests failed - Please review errors above")
        return 1


if __name__ == '__main__':
    sys.exit(main())
