"""
Common utilities for Image Processing & Computer Vision assignments.
Includes image loading, saving, display, and metric computation functions.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os


def load_image(image_path, grayscale=False):
    """
    Load an image from disk.
    
    Args:
        image_path: Path to the image file
        grayscale: If True, convert to grayscale; else load as color
    
    Returns:
        Image array (H, W, 3) for color or (H, W) for grayscale
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img


def save_image(image_array, output_path):
    """
    Save an image array to disk.
    
    Args:
        image_array: Image array to save
        output_path: Path where to save the image
    
    Returns:
        True if successful, False otherwise
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    success = cv2.imwrite(output_path, image_array)
    return success


def resize_image(image, size=(512, 512)):
    """
    Resize image to specified dimensions.
    
    Args:
        image: Input image array
        size: Target size as (width, height)
    
    Returns:
        Resized image
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def convert_to_grayscale(image):
    """Convert color image to grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def compute_mse(image1, image2):
    """
    Compute Mean Squared Error between two images.
    
    Args:
        image1, image2: Images to compare (same shape)
    
    Returns:
        MSE value
    """
    image1 = image1.astype(np.float32)
    image2 = image2.astype(np.float32)
    mse = np.mean((image1 - image2) ** 2)
    return mse


def compute_psnr(image1, image2, max_val=255.0):
    """
    Compute Peak Signal-to-Noise Ratio between two images.
    
    Args:
        image1, image2: Images to compare
        max_val: Maximum pixel value (default 255 for 8-bit)
    
    Returns:
        PSNR value in dB
    """
    mse = compute_mse(image1, image2)
    if mse == 0:
        return 100.0
    psnr = 20 * np.log10(max_val / np.sqrt(mse))
    return psnr


def compute_ssim(image1, image2):
    """
    Compute Structural Similarity Index Measure (SSIM).
    
    Args:
        image1, image2: Images to compare (grayscale)
    
    Returns:
        SSIM value (between -1 and 1, where 1 is identical)
    """
    if len(image1.shape) > 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) > 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    image1 = image1.astype(np.float32) / 255.0
    image2 = image2.astype(np.float32) / 255.0
    
    score, _ = ssim(image1, image2, full=True, data_range=1.0)
    return score


def display_images(images, titles, figsize=(15, 10), save_path=None):
    """
    Display multiple images in a grid.
    
    Args:
        images: List of image arrays
        titles: List of titles for each image
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
    """
    n_images = len(images)
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    plt.figure(figsize=figsize)
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, idx + 1)
        
        # Handle different image formats
        if len(img.shape) == 2:  # Grayscale
            plt.imshow(img, cmap='gray')
        else:  # Color (BGR -> RGB)
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        plt.title(title, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return plt.gcf()


def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=1.0):
    """Apply Gaussian blur to image."""
    return cv2.GaussianBlur(image, kernel_size, sigma)


def apply_median_filter(image, kernel_size=5):
    """Apply median filter to image."""
    return cv2.medianBlur(image, kernel_size)


def apply_mean_filter(image, kernel_size=5):
    """Apply mean (box) filter to image."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def add_gaussian_noise(image, mean=0, std=25):
    """Add Gaussian noise to image."""
    noise = np.random.normal(mean, std, image.shape)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def add_salt_pepper_noise(image, prob=0.05):
    """Add salt-and-pepper noise to image."""
    noisy = image.copy().astype(np.float32)
    mask = np.random.rand(*image.shape)
    
    # Salt
    noisy[mask < prob/2] = 255
    # Pepper
    noisy[mask > 1 - prob/2] = 0
    
    return noisy.astype(np.uint8)


def downsample_image(image, factor):
    """Downsample image by specified factor."""
    h, w = image.shape[:2]
    new_h, new_w = h // factor, w // factor
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def upsample_image(image, target_shape):
    """Upsample image to target shape."""
    return cv2.resize(image, target_shape, interpolation=cv2.INTER_LINEAR)


def quantize_image(image, levels):
    """
    Quantize image to specified number of gray levels.
    
    Args:
        image: Input grayscale image
        levels: Number of gray levels to quantize to
    
    Returns:
        Quantized image
    """
    quantized = (image.astype(np.float32) / 255.0 * (levels - 1)).round()
    quantized = (quantized / (levels - 1) * 255.0).astype(np.uint8)
    return quantized


def histogram_equalization(image):
    """Apply histogram equalization to enhance contrast."""
    if len(image.shape) == 3:
        # Convert to LAB color space, equalize L channel, convert back
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        return cv2.equalizeHist(image)


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)


def print_section_header(title):
    """Print a formatted section header."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title.center(width - 4)}")
    print("=" * width)


def print_metrics(name, mse, psnr, ssim_score=None):
    """Print image quality metrics in a formatted way."""
    print(f"\n{name}:")
    print(f"  MSE  : {mse:.4f}")
    print(f"  PSNR : {psnr:.4f} dB")
    if ssim_score is not None:
        print(f"  SSIM : {ssim_score:.4f}")
