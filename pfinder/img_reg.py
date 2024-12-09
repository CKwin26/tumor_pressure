
# -*- coding: utf-8 -*-
"""
Image Registration and Processing Script for Pfinder Module

This script performs image registration and processing as part of the Pfinder module.
It aligns a moving image to a reference image using ORB feature matching, crops the aligned image, 
and prepares the output for downstream analysis such as bead displacement and pressure calculations.

FUNCTIONALITY:
1. Load and preprocess images:
   - Enhances image contrast using CLAHE.
   - Masks the center of the image to focus on features for alignment.
2. Image registration:
   - Uses ORB (Oriented FAST and Rotated BRIEF) to detect keypoints.
   - Matches keypoints and calculates an affine transformation matrix.
   - Aligns the moving image to the reference image.
3. Image cropping:
   - Crops both reference and aligned images to a specified region of interest (ROI).
4. Save and visualize results:
   - Saves the processed images to the output directory.
   - Displays the cropped reference and aligned images for verification.

INPUTS:
    
1. `image_paths`: List of two image file paths:
   - Path to the reference image (first in the list).
   - Path to the moving image (second in the list).
2. `output_dir`: Path to the directory where the processed images will be saved.
3. `crop_size`: User-defined ratio for cropping (value between `0` and `0.99`):
   - For example, `0.8` retains 80% of the original image width and height, cropping 10% from each side.

OUTPUTS:
- Saves the aligned and cropped image to the specified output directory.
- Displays the processed images (reference and aligned).

DEPENDENCIES:
- OpenCV (cv2): For image processing and alignment.
- NumPy: For numerical operations.
- Matplotlib: For visualizing results.
- ImageIO: For loading and saving images.
- OS: For handling file paths and directories.

WORKFLOW INTEGRATION:
- Prepares processed images for bead displacement analysis.
- Outputs can be used in subsequent Pfinder steps such as bead selection and pressure calculation.


Created on Mon Oct 7, 2024
Author: Kowsh
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
import logging

def load_images(paths):
    images = [imageio.imread(path) for path in paths]
    return images

def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(image)
    return enhanced_image

def mask_center(image, mask_size):
    h, w = image.shape
    center_x, center_y = w // 2, h // 2
    half_mask_size = mask_size // 2
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask[center_y - half_mask_size:center_y + half_mask_size, center_x - half_mask_size:center_x + half_mask_size] = 0
    return cv2.bitwise_and(image, mask)

def align_images_orb(reference_image, moving_image, mask_size):
    orb = cv2.ORB_create()
    reference_image_enhanced = enhance_contrast(reference_image)
    moving_image_enhanced = enhance_contrast(moving_image)
    reference_image_masked = mask_center(reference_image_enhanced, mask_size)
    moving_image_masked = mask_center(moving_image_enhanced, mask_size)
    keypoints1, descriptors1 = orb.detectAndCompute(reference_image_masked, None)
    keypoints2, descriptors2 = orb.detectAndCompute(moving_image_masked, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    M, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts)
    aligned_image = cv2.warpAffine(moving_image, M, (reference_image.shape[1], reference_image.shape[0]))
    return aligned_image, reference_image

def crop_image(image, x_start, x_end, y_start, y_end):
    return image[y_start:y_end, x_start:x_end]

def main(image_paths, output_dir, crop_size):
    """
    Main function for image registration and cropping.
    """
    logging.info("Starting image registration...")
    
    if not image_paths or len(image_paths) != 2:
        raise ValueError("Two image paths (reference and moving) must be provided.")
    
    if not output_dir:
        raise ValueError("Output directory must be provided.")
    
    if not (0 <= crop_size < 1):
        raise ValueError("Crop size must be between 0 and 1.")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Input images: {image_paths}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Crop size: {crop_size}")

    try:
        # Load images
        images = load_images(image_paths)
        reference_image, moving_image = images[0], images[1]

        # Align images
        aligned_image, _ = align_images_orb(reference_image, moving_image, mask_size=1000)

        # Crop images
        h, w = reference_image.shape
        x_start, x_end = int(w * crop_size), w - int(w * crop_size)
        y_start, y_end = int(h * crop_size), h - int(h * crop_size)

        reference_image_cropped = crop_image(reference_image, x_start, x_end, y_start, y_end)
        aligned_image_cropped = crop_image(aligned_image, x_start, x_end, y_start, y_end)

        # Save images
        ref_output_path = os.path.join(output_dir, "ref_cropped.tif")
        aligned_output_path = os.path.join(output_dir, "aligned_cropped.tif")

        imageio.imwrite(ref_output_path, reference_image_cropped)
        imageio.imwrite(aligned_output_path, aligned_image_cropped)

        logging.info(f"Cropped reference image saved to {ref_output_path}.")
        logging.info(f"Aligned and cropped image saved to {aligned_output_path}.")

        # Display images
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(reference_image_cropped, cmap='gray')
        axs[0].set_title('Cropped Reference Image')
        axs[1].imshow(aligned_image_cropped, cmap='gray')
        axs[1].set_title('Aligned and Cropped Image')
        plt.show()

        return aligned_image_cropped, reference_image_cropped

    except Exception as e:
        logging.error(f"Error during image registration: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # 示例调用
    sample_image_paths = ["path_to_reference_image.tif", "path_to_moving_image.tif"]
    sample_output_dir = "path_to_output_directory"
    sample_crop_size = 0.2

    main(sample_image_paths, sample_output_dir, sample_crop_size)








