# Image Processing & Computer Vision - Five Mini Projects

**Course:** Image Processing & Computer Vision  
**Student Name:** Aditya Raj Singh  
**Roll No:** 2301010195  
**Date:** April 29, 2026  

---

## Project Overview

This repository contains **five integrated mini-projects** that comprehensively cover Image Processing and Computer Vision fundamentals. Each assignment addresses real-world problems using Python with OpenCV, NumPy, and SciPy.

### Projects Summary

| # | Title | Learning Focus | Key Techniques |
|---|-------|-----------------|-----------------|
| **1** | Smart Document Scanner | Sampling & Quantization | Resolution analysis, bit-depth reduction |
| **2** | Image Restoration | Noise & Filtering | Gaussian/Salt-Pepper noise, Mean/Median/Gaussian filters |
| **3** | Medical Imaging System | Compression & Segmentation | Run-Length Encoding, Thresholding, Morphology |
| **4** | Traffic Monitoring | Feature Extraction | Edge detection, Contours, ORB features |
| **5** | Capstone: Intelligent System | End-to-End Pipeline | Integration of all techniques (Units 1-4) |

---

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup Instructions

1. **Clone or navigate to the project directory:**
   ```bash
   cd image_projects
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate          # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies
- `opencv-python` - Image processing
- `numpy` - Numerical computations
- `matplotlib` - Visualization
- `scikit-image` - Advanced image processing
- `scipy` - Scientific computing
- `Pillow` - Image I/O
- `imutils` - OpenCV utilities

---

## Project Structure

```
image_projects/
├── lab-experiments/            # Lab experiment scripts (run individually)
│   ├── asmt1_scanner/          # Assignment 1: Document Scanning
│   │   └── scanner.py          # Main script - document quality analysis
│   ├── asmt2_restoration/      # Assignment 2: Image Restoration
│   │   └── restoration.py      # Main script - noise filtering
│   ├── asmt3_medical/          # Assignment 3: Medical Imaging
│   │   └── medical_image_system.py # Main script - compression & segmentation
│   ├── asmt4_traffic/          # Assignment 4: Traffic Monitoring
│   │   └── traffic_monitoring.py   # Main script - feature extraction
│   ├── asmt5_intelligent_system/   # Assignment 5: Capstone
│   │   └── main.py             # Main capstone script - end-to-end pipeline
│   └── common/
│       └── utils.py            # Shared utility functions
├── assets/                     # Sample images (download manually)
│   ├── asmt1/                  # Document images
│   ├── asmt2/                  # Surveillance/street images
│   ├── asmt3/                  # Medical/X-ray images
│   ├── asmt4/                  # Traffic/intersection images
│   └── asmt5/                  # General purpose images
├── outputs/                    # Generated results (auto-created)
├── requirements.txt            # Dependencies
├── run_all.py                  # Test harness (runs all 5 assignments)
├── cleanup_outputs.py          # Output cleanup utility
└── README.md                   # This file
```

### Lab Experiments Directory Structure
The `lab-experiments/` directory contains all individual assignment scripts. Each can be run independently:
- Each assignment is self-contained with its own directory
- All scripts share utilities from `common/utils.py`
- Run any assignment individually with custom images
- Use `run_all.py` to batch test all assignments

---

## Usage Instructions

### Running Individual Assignments

#### Assignment 1: Document Scanner
```bash
python asmt1_scanner/scanner.py \
  --input assets/asmt1/sample_1.jpg \
  --output outputs/asmt1
```

#### Assignment 2: Image Restoration
```bash
python asmt2_restoration/restoration.py \
  --input assets/asmt2/sample_1.jpg \
  --output outputs/asmt2
```

#### Assignment 3: Medical Image System
```bash
python asmt3_medical/medical_image_system.py \
  --input assets/asmt3/sample_1.jpg \
  --output outputs/asmt3
```

#### Assignment 4: Traffic Monitoring
```bash
python asmt4_traffic/traffic_monitoring.py \
  --input assets/asmt4/sample_1.jpg \
  --output outputs/asmt4
```

#### Assignment 5: Capstone System
```bash
python asmt5_intelligent_system/main.py \
  --input assets/asmt5/sample_1.jpg \
  --output outputs/asmt5
```

### Running All Assignments (Test Harness)
```bash
python run_all.py
```

This will:
- Execute each assignment on all sample images (3 per assignment)
- Generate comprehensive output figures
- Produce metrics and analysis reports
- Create a summary of all results

### Command-Line Arguments

All scripts support:
```
--input, -i     : Path to input image (default: auto-select from assets/)
--output, -o    : Output directory for results (default: outputs/asmtX/)
--help, -h      : Show help message
```

---

## Output Files Generated

Each assignment generates:
- **Output Images:** Processing results (filtering, segmentation, features, etc.)
- **Comparison Figures:** Multi-panel visualizations of stages
- **Metrics Files:** Quantitative evaluation (MSE, PSNR, SSIM)
- **Console Output:** Detailed analysis and observations

### Example Outputs
```
outputs/
├── asmt1/
│   ├── sampling_512x512.jpg
│   ├── sampling_256x256.jpg
│   ├── sampling_128x128.jpg
│   ├── quantized_256_levels_8bit.jpg
│   ├── quantized_16_levels_4bit.jpg
│   ├── quantized_4_levels_2bit.jpg
│   ├── sampling_comparison.png
│   └── quantization_comparison.png
├── asmt2/
│   ├── noisy_gaussian.jpg
│   ├── noisy_salt_pepper.jpg
│   ├── restored_gaussian_mean.jpg
│   ├── restored_gaussian_median.jpg
│   ├── restored_gaussian_gaussian.jpg
│   └── restoration_comparison.png
├── asmt3/
│   ├── segmented_global_threshold.jpg
│   ├── segmented_otsu_threshold.jpg
│   ├── morpho_global_opened.jpg
│   ├── morpho_otsu_closed.jpg
│   └── segmentation_comparison.png
├── asmt4/
│   ├── edges_sobel.jpg
│   ├── edges_canny.jpg
│   ├── contours_sobel.jpg
│   ├── bounding_boxes_canny.jpg
│   ├── features_orb_keypoints.jpg
│   └── edge_detection_comparison.png
└── asmt5/
    ├── stage_enhancement_restoration.png
    ├── stage_segmentation_morphology.png
    ├── stage_feature_extraction.png
    ├── complete_pipeline.png
    └── metrics.txt
```

---

## Key Algorithms & Techniques

### Assignment 1: Document Scanning
- **Grayscale Conversion:** BGR to grayscale transformation
- **Sampling:** Downsampling with nearest-neighbor interpolation
- **Quantization:** Bit-depth reduction (256 → 16 → 4 → 2 levels)
- **Quality Metrics:** MSE, PSNR comparison

### Assignment 2: Image Restoration
- **Noise Models:** Gaussian noise (σ=25), Salt-and-Pepper (p=0.05)
- **Filters:** Mean (kernel=5), Median (kernel=5), Gaussian (σ=1)
- **Evaluation:** MSE, PSNR for each filter type
- **Recommendation:** Median best for impulse noise; Gaussian for additive

### Assignment 3: Medical Imaging
- **Compression:** Run-Length Encoding (RLE) for lossless compression
- **Segmentation:** Global thresholding, Otsu's adaptive threshold
- **Morphology:** Erosion, Dilation, Opening, Closing
- **Analysis:** Compression ratio, storage savings, clinical relevance

### Assignment 4: Traffic Monitoring
- **Edge Detection:** Sobel operator, Canny edge detector
- **Object Representation:** Contour detection, bounding boxes
- **Feature Extraction:** ORB (Oriented FAST & Rotated BRIEF)
- **Comparative Analysis:** Filter performance, use cases

### Assignment 5: Capstone
- **Complete Pipeline:** Integrates all previous techniques
- **Enhancement:** Histogram equalization, CLAHE
- **Restoration:** Multi-filter approach (Median + Gaussian)
- **Quality Metrics:** MSE, PSNR, SSIM
- **End-to-End Workflow:** Acquisition → Enhancement → Segmentation → Feature extraction

---

## Sample Images & Assets

### Obtaining Sample Images

You need to download **15 sample images** (3 per assignment) and organize them in the `assets/` directory. Choose from royalty-free image sources like Unsplash, Pexels, or Pixabay.

#### Asset Requirements by Assignment

**Assignment 1: Document Scanner (asmt1/)**
- Type: Document or text-heavy images
- Best sources: Scan/photo of documents, textbooks, papers, invoices
- Size: 512×512 or larger
- Count: 3 images
- Files: `sample_1.jpg`, `sample_2.jpg`, `sample_3.jpg`

**Assignment 2: Image Restoration (asmt2/)**
- Type: Surveillance or outdoor scenes
- Best sources: Street scenes, traffic, buildings, landscapes
- Size: 512×512 or larger
- Count: 3 images
- Files: `sample_1.jpg`, `sample_2.jpg`, `sample_3.jpg`

**Assignment 3: Medical Imaging (asmt3/)**
- Type: Medical or X-ray-like images
- Best sources: Medical textures, bone structures, or high-contrast images
- Size: 512×512 or larger
- Count: 3 images
- Files: `sample_1.jpg`, `sample_2.jpg`, `sample_3.jpg`

**Assignment 4: Traffic Monitoring (asmt4/)**
- Type: Traffic scenes with clear object boundaries
- Best sources: Traffic intersections, road images, vehicle scenes
- Size: 512×512 or larger
- Count: 3 images
- Files: `sample_1.jpg`, `sample_2.jpg`, `sample_3.jpg`

**Assignment 5: Capstone (asmt5/)**
- Type: General purpose mixed scenes
- Best sources: Any natural or urban scenes with variety
- Size: 512×512 or larger
- Count: 3 images
- Files: `sample_1.jpg`, `sample_2.jpg`, `sample_3.jpg`

### Directory Structure for Assets

Create the `assets/` folder with this structure before running scripts:
```
assets/
├── asmt1/
│   ├── sample_1.jpg
│   ├── sample_2.jpg
│   └── sample_3.jpg
├── asmt2/
│   ├── sample_1.jpg
│   ├── sample_2.jpg
│   └── sample_3.jpg
├── asmt3/
│   ├── sample_1.jpg
│   ├── sample_2.jpg
│   └── sample_3.jpg
├── asmt4/
│   ├── sample_1.jpg
│   ├── sample_2.jpg
│   └── sample_3.jpg
└── asmt5/
    ├── sample_1.jpg
    ├── sample_2.jpg
    └── sample_3.jpg
```

**Total images needed: 15 (3 per assignment)**

**Recommended sources (CC0/free):**
- [Unsplash](https://unsplash.com/)
- [Pexels](https://www.pexels.com/)
- [Pixabay](https://pixabay.com/)
- [Freepik](https://www.freepik.com/)

### Using Custom Images
To use your own images:
```bash
python asmt1_scanner/scanner.py --input path/to/your/image.jpg --output custom_outputs/
```

---

## Image Quality Metrics

### Mean Squared Error (MSE)
$$\text{MSE} = \frac{1}{MN} \sum_{i=0}^{M-1} \sum_{j=0}^{N-1} (I_{original}(i,j) - I_{processed}(i,j))^2$$

Lower is better. Range: [0, ∞)

### Peak Signal-to-Noise Ratio (PSNR)
$$\text{PSNR} = 20 \log_{10} \left( \frac{\text{MAX}}{\sqrt{\text{MSE}}} \right)$$

Higher is better. Typical range: [20, 50] dB

### Structural Similarity Index (SSIM)
$$\text{SSIM} = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$

Range: [-1, 1]. Higher indicates more similar images.

---


## References & Resources

### Official Documentation
- [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/users/index.html)
- [scikit-image Documentation](https://scikit-image.org/)

### Textbooks
- **Gonzalez & Woods** - "Digital Image Processing" (3rd Edition)
- **Forsyth & Ponce** - "Computer Vision: A Modern Approach"
- **Szeliski** - "Computer Vision: Algorithms and Applications"

### Algorithms
- Canny Edge Detection: Canny, J. (1986). "A Computational Approach to Edge Detection"
- ORB Features: Rublee et al. (2011). "ORB: An Efficient Alternative to SIFT or SURF"
- Otsu Thresholding: Otsu, N. (1979). "A Threshold Selection Method from Gray-Level Histograms"

### Noise & Filtering
- Gaussian Blur: Blur image using Gaussian kernel
- Median Filter: Edge-preserving non-linear filter for impulse noise
- Run-Length Encoding: Lossless compression for runs of identical values

---

## Cleaning Up Output Files

### Using the Cleanup Script

The project includes a utility script to safely remove all generated output files while preserving the directory structure for future runs.

#### Basic Usage (Interactive Mode)
```bash
python cleanup_outputs.py
```

The script will:
- Display current output directory statistics
- Show number of files and total size to be removed
- Ask for confirmation before deleting
- Remove all files and report success

#### Force Cleanup (No Confirmation)
```bash
python cleanup_outputs.py --force
```

or

```bash
python cleanup_outputs.py -f
```

#### What Gets Cleaned
- All image outputs (JPG, PNG)
- All generated figures and comparisons
- All metrics and analysis files
- Directory structure is **preserved** for future runs

#### What is NOT Cleaned
- Source code remains intact
- Configuration files remain unchanged
- The `assets/` folder is not affected
- `requirements.txt` and documentation remain unchanged

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'cv2'"
**Solution:**
```bash
pip install opencv-python
```

### Issue: "FileNotFoundError: Image not found"
**Solution:**
```bash
# Generate sample images first
python assets/generate_images.py

# Or specify full path to your image
python asmt1_scanner/scanner.py --input /full/path/to/image.jpg
```

### Issue: Out of memory with large images
**Solution:**
- Reduce image size in preprocessing (already set to 512×512)
- Close other applications
- Process one image at a time

### Issue: Display backend error in headless environment
**Solution:**
```python
# Add to script before matplotlib imports
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
```

---

## Performance Notes

### Expected Execution Times
- **Assignment 1** (Scanner): ~5-10 seconds per image
- **Assignment 2** (Restoration): ~10-15 seconds per image
- **Assignment 3** (Medical): ~5-10 seconds per image
- **Assignment 4** (Traffic): ~8-12 seconds per image
- **Assignment 5** (Capstone): ~30-60 seconds per image

### System Requirements
- **RAM:** Minimum 2GB (recommended 4GB+)
- **Storage:** ~500MB for complete project with outputs
- **Processor:** Dual-core minimum (quad-core recommended)

---

## Future Enhancements

Possible extensions for advanced learning:
1. Real-time video processing (Assignment 4)
2. GPU acceleration using CUDA/OpenCL
3. Deep learning-based segmentation (Segment Anything, U-Net)
4. Multi-scale feature extraction (Scale-Invariant Feature Transform)
5. Object tracking and trajectory analysis
6. Web interface for interactive processing

---

## Contact & Support

**Course Instructor:** satinder.palsingh@krmangalam.edu.in  
**Course:** Image Processing & Computer Vision  
**Institution:** K.R. Mangalam University  

---

**Last Updated:** April 28, 2026  
**Status:** Complete - Ready for Submission

---

## Checklist for Submission

- [x] All 5 scripts implemented and tested
- [x] Sample images downloaded and organized in `assets/` folder (3 per assignment)
- [x] Requirements.txt updated and verified
- [x] All outputs generated and saved in `outputs/` folder
- [x] README.md with complete documentation
- [x] run_all.py test harness working (verify with `python run_all.py`)
- [x] Code comments and headers added
#   i p c v - c o l l e g e - w o r k - m a i n  
 