#!/usr/bin/env python3
"""
Cleanup script to remove output files generated during assignments.

This script safely removes all files and folders within the outputs/ directory
while preserving the directory structure for future runs.

Usage:
    python cleanup_outputs.py              # Interactive mode
    python cleanup_outputs.py --force      # Force cleanup without confirmation
"""

import os
import shutil
import sys
from pathlib import Path


def get_output_dir():
    """Get the outputs directory path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'outputs')


def get_directory_size(path):
    """Calculate total size of directory in bytes."""
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                if os.path.exists(filepath):
                    total += os.path.getsize(filepath)
    except Exception as e:
        print(f"Error calculating size: {e}")
    return total


def format_size(bytes_size):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024
    return f"{bytes_size:.2f} TB"


def count_files(path):
    """Count total files in directory."""
    count = 0
    for dirpath, dirnames, filenames in os.walk(path):
        count += len(filenames)
    return count


def cleanup_outputs(force=False):
    """Remove all output files while preserving directory structure."""
    output_dir = get_output_dir()
    
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return False
    
    # Get statistics before cleanup
    file_count = count_files(output_dir)
    total_size = get_directory_size(output_dir)
    
    if file_count == 0:
        print("Output directory is already clean (no files found).")
        return True
    
    print("\n" + "="*70)
    print("OUTPUT DIRECTORY CLEANUP")
    print("="*70)
    print(f"\nDirectory: {output_dir}")
    print(f"Files to remove: {file_count}")
    print(f"Space to free: {format_size(total_size)}")
    
    if not force:
        print("\n[WARNING] This will delete all output files. This action cannot be undone.")
        response = input("\nAre you sure you want to proceed? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("Cleanup cancelled.")
            return False
    
    # Remove all files while keeping directory structure
    try:
        removed_count = 0
        for dirpath, dirnames, filenames in os.walk(output_dir):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    os.remove(filepath)
                    removed_count += 1
                except Exception as e:
                    print(f"Warning: Could not remove {filepath}: {e}")
        
        print(f"\n[OK] Successfully removed {removed_count} files")
        print(f"[OK] Freed: {format_size(total_size)}")
        print("[OK] Directory structure preserved for future runs")
        return True
        
    except Exception as e:
        print(f"\n[FAIL] Error during cleanup: {e}")
        return False


def main():
    """Main entry point."""
    force_mode = '--force' in sys.argv or '-f' in sys.argv
    
    if cleanup_outputs(force=force_mode):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
