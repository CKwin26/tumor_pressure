
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
def load_images(paths):
    images = [imageio.imread(path) for path in paths]
    return images
def enhance_contrast(image):
    # Enhance contrast to capture characteristics
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
    # Create ORB feature detector
    orb = cv2.ORB_create()
    # Enhance contrast
    reference_image_enhanced = enhance_contrast(reference_image)
    moving_image_enhanced = enhance_contrast(moving_image)
    # Apply mask to the images
    reference_image_masked = mask_center(reference_image_enhanced, mask_size)
    moving_image_masked = mask_center(moving_image_enhanced, mask_size)
    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(reference_image_masked, None)
    keypoints2, descriptors2 = orb.detectAndCompute(moving_image_masked, None)
    # Create BFMatcher for descriptor matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    # Extract matched point positions
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    # Compute affine transform matrix
    M, mask = cv2.estimateAffinePartial2D(dst_pts, src_pts)
    # Apply affine transform to align images
    aligned_image = cv2.warpAffine(moving_image, M, (reference_image.shape[1], reference_image.shape[0]))
    return aligned_image, reference_image



def crop_image(image, x_start, x_end, y_start, y_end):
    return image[y_start:y_end, x_start:x_end]



import logging

def main(image_paths=None, output_dir=None, crop_size=None):
    logging.info("Starting image registration...")

    if image_paths is None:
        logging.info("Prompting user for image paths.")
        print("Enter the full paths to the images(reference first and then moving image), separated by a comma:")
        image_paths = input().strip().split(',')
        image_paths = [path.strip() for path in image_paths]

    if output_dir is None:
        logging.info("Prompting user for output directory.")
        print("Enter the output directory path:")
        output_dir = input().strip()

    if crop_size is None:
        logging.info("Prompting user for crop size.")
        crop_size = float(input("How much you want to crop the image? input 0 to 0.99: ").strip().lower())

    logging.info(f"Input images: {image_paths}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Crop size: {crop_size}")

    try:
        # 调用实际逻辑
        images = load_images(image_paths)
        aligned_image, reference_image_cropped = align_images_orb(images[0], images[1], mask_size=1000)
        h, w = reference_image_cropped.shape
        print(h,w)
        cropped_image = crop_image(aligned_image, int(w*crop_size), w-int(w*crop_size), int(h*crop_size), h-int(h*crop_size))
        print(int(w*crop_size))
        print(w-int(w*crop_size))
        # 保存和日志记录
        output_path = os.path.join(output_dir, "aligned_and_cropped.tif")
        imageio.imwrite(output_path, cropped_image)
        logging.info(f"Aligned and cropped image saved to {output_path}.")
    except Exception as e:
        logging.error(f"Error during image registration: {str(e)}", exc_info=True)
        
    
    images = load_images(image_paths)
    reference_image = cv2.convertScaleAbs(images[0])
    moving_image = cv2.convertScaleAbs(images[1])
    # Define mask size
    mask_size = 1000
    # Align images
    aligned_image, reference_image_masked = align_images_orb(reference_image, moving_image, mask_size)
    # Crop images to specified dimensions
    h, w = reference_image.shape 
    x_start= int(w*crop_size)
    x_end= w-int(w*crop_size)
    y_start= int(h*crop_size)
    y_end= h-int(h*crop_size)
    
    reference_image_cropped = crop_image(reference_image, x_start, x_end, y_start, y_end)
    aligned_image_cropped = crop_image(aligned_image, x_start, x_end, y_start, y_end)
    # Save images
    #imageio.imwrite(os.path.join(output_dir, '0ref.tiff'), reference_image_cropped)
    imageio.imwrite(os.path.join(output_dir), aligned_image_cropped)
    print("Aligned and cropped image saved.")
    print("Cropped reference image saved.")
    # Display aligned and cropped image and masked reference image
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(reference_image_cropped, cmap='gray')
    axs[0].set_title('Cropped Reference Image')
    axs[1].imshow(aligned_image_cropped, cmap='gray')
    axs[1].set_title('Aligned and Cropped Image')
    plt.show()
    return aligned_image_cropped,reference_image_cropped
 

if __name__ == "__main__":
    
   
    main()







