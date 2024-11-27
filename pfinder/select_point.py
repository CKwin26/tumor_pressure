"""
Manual Keypoint Selection and Image Processing Script

This script is designed to assist in manually selecting keypoints from a reference and moving image for alignment purposes.
It preprocesses images to enhance contrast, applies masks, detects contours, and calculates centroids to guide the manual selection process. 
The manually selected keypoints are saved to an Excel file for downstream analysis.

FUNCTIONALITY:
1. Image Loading and Resizing:
   - Loads the reference and moving images from specified paths.
   - Resizes images for faster processing and better visualization.

2. Image Preprocessing:
   - Enhances contrast using CLAHE for better keypoint visualization.
   - Applies circular masks to isolate regions of interest.
   - Detects the largest contours in the masked images and computes centroids.

3. Manual Keypoint Selection:
   - Provides an interactive interface for selecting corresponding keypoints on the reference and moving images.
   - Displays selected keypoints in real-time and allows deletion using a right-click.

4. Data Saving:
   - Saves the manually selected keypoints to an Excel file.
   - Includes coordinates of each keypoint pair along with their match index.

5. Visualization:
   - Displays the reference and moving images with annotated keypoints.

INPUTS:
1. `image_paths`: List of two image file paths:
   - First image is the reference image.
   - Second image is the moving image.
2. `outd`: Path to the output directory where results will be saved.
3. `excel_name`: Name of the Excel file where the matched points will be saved.

OUTPUTS:
1. Excel File:
   - Contains the coordinates of manually selected keypoints from both images:
     - `Match Index`: Index of each matched keypoint pair.
     - `x1, y1`: Coordinates in the reference image.
     - `x2, y2`: Coordinates in the moving image.

2. Visualization:
   - Displays the reference and moving images with annotated keypoints for verification.

DEPENDENCIES:
- OpenCV (cv2): For image processing and visualization.
- NumPy: For numerical operations.
- Matplotlib: For plotting and visualization.
- Scikit-image: For image filtering and morphology operations.
- Pandas: For saving keypoints to an Excel file.
- ImageIO: For reading and saving images.

WORKFLOW:
1. Load and preprocess images:
   - Resize images for faster processing.
   - Enhance contrast and apply circular masks.
   - Detect contours and calculate centroids for guidance.

2. Perform manual keypoint selection:
   - Left-click to select keypoints on both images.
   - Right-click to delete keypoints.
   - Ensure the number of keypoints is consistent across both images.

3. Save results:
   - Saves keypoints in an Excel file with the specified name.
   - Annotated images with keypoints can also be saved for verification.


"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import pandas as pd
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import binary_closing, disk, remove_small_objects, label
from scipy.ndimage.measurements import center_of_mass
scale=0.5
def load_images(paths):
    images = [resize_image(imageio.imread(path), scale=scale) for path in paths]
    return images

def resize_image(image, scale=0.5, width=None, height=None):
    (h, w) = image.shape[:2]
    if scale:
        new_dim = (int(w * scale), int(h * scale))
    elif width and height:
        new_dim = (width, height)
    elif width:
        scale = width / float(w)
        new_dim = (width, int(h * scale))
    elif height:
        scale = height / float(h)
        new_dim = (int(w * scale), height)
    else:
        raise ValueError("You must provide either scale, width, or height.")
    
    resized = cv2.resize(image, new_dim, interpolation=cv2.INTER_AREA)
    return resized

def create_circle_mask(img, radius=500):
    height, width = img.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    center_x, center_y = width // 2, height // 2
    cv2.circle(mask, (center_x, center_y), radius, 255, -1)
    masked_image = cv2.bitwise_and(img, img, mask=mask)
    return mask, masked_image

def enhance_contrast(image, clip_limit=5.0):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(27, 27))
    enhanced_image = clahe.apply(image)
    return enhanced_image

def adjust_threshold(image, threshold_scale):
    otsu_thresh = threshold_otsu(image)
    adjusted_thresh = otsu_thresh * threshold_scale
    binary_image = image < adjusted_thresh  # Invert the binary mask
    return binary_image

def create_largest_contour_mask(image, threshold_scale=1.3):
    def adjust_threshold(image, threshold_scale):
        otsu_thresh = threshold_otsu(image)
        adjusted_thresh = otsu_thresh * threshold_scale
        binary_image = image < adjusted_thresh  # Invert the binary mask
        return binary_image

    blurred_image = gaussian(image, sigma=2)
    adjusted_binary = adjust_threshold(blurred_image, threshold_scale)
    adjusted_binary = remove_small_objects(adjusted_binary, min_size=1000)
    adjusted_binary = binary_closing(adjusted_binary, selem=disk(5))
    adjusted_binary_uint8 = (adjusted_binary * 255).astype(np.uint8)
    contours, _ = cv2.findContours(adjusted_binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(adjusted_binary_uint8)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], 0, (255), thickness=cv2.FILLED)
        print(len(contours))
    return mask

def compute_centroid(binary_mask):
    labeled_mask, num_labels = label(binary_mask, return_num=True)
    largest_region = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1
    largest_region_mask = labeled_mask == largest_region
    centroid = center_of_mass(largest_region_mask)
    return int(centroid[1]), int(centroid[0])  # Return (x, y)

def apply_mask(original_image, binary_mask):
    masked_image = np.where(binary_mask, 0, original_image)  # Set mask area to 0 (black), keep other areas unchanged
    return masked_image

def plot_mask_with_centroid(image, mask, centroid, title, output_path):
    # 创建彩色掩码
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored_mask[mask == 255] = [255, 0, 0]  # 将掩码部分设为红色

    # 将掩码叠加在原图上
    overlay = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 0.7, colored_mask, 0.3, 0)

    # 绘制质心
    plt.figure(figsize=(10, 10))
    plt.imshow(overlay)
    plt.plot(centroid[0], centroid[1], 'ro')  # 用红点标记质心
    plt.title(title)
    plt.savefig(output_path)
    plt.show()

def find_min_grayvalue_point(image, x, y, box_size=10):
    x_min = max(0, x - box_size // 2)
    x_max = min(image.shape[1], x + box_size // 2 + 1)
    y_min = max(0, y - box_size // 2)
    y_max = min(image.shape[0], y + box_size // 2 + 1)

    min_val = np.min(image[y_min:y_max, x_min:x_max])
    min_loc = np.where(image[y_min:y_max, x_min:x_max] == min_val)
    min_x = x_min + min_loc[1][0]
    min_y = y_min + min_loc[0][0]
    
    return (min_x, min_y)





def manual_select_keypoints(images, center, titles=["Reference Image", "Moving Image"], scale=0.5):
    keypoints = [[], []]
    selected_point = [-1, -1]
    images = [resize_image(image, scale) for image in images]
    
    def select_point(event, x, y, flags, param):
        img_index = param["img_index"]
        if event == cv2.EVENT_LBUTTONDOWN:
            min_point = find_min_grayvalue_point(images[img_index], x, y)
            keypoints[img_index].append(min_point)
            
            cv2.circle(images[img_index], min_point, 5, (0, 255, 0), -1)
            cv2.putText(images[img_index], f"{len(keypoints[img_index])}", min_point, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(titles[img_index], images[img_index])
            if img_index == 0:
                selected_point[0] = min_point
                draw_rectangle_on_other_image(*min_point)
            elif img_index == 1:
                selected_point[1] = min_point
                clear_rectangle_on_other_image()
        elif event == cv2.EVENT_RBUTTONDOWN:
            for i, (px, py) in enumerate(keypoints[img_index]):
                if (px - 10 <= x <= px + 10) and (py - 10 <= y <= py + 10):
                    keypoints[img_index].pop(i)
                    if len(keypoints[1 - img_index]) > i:
                        keypoints[1 - img_index].pop(i)
                    redraw_images()
                    return

    def draw_rectangle_on_other_image(x, y):
        x_min = max(0, x - 10)
        x_max = min(images[1].shape[1], x + 10)
        y_min = max(0, y - 10)
        y_max = min(images[1].shape[0], y + 10)
        cv2.rectangle(images[1], (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.imshow(titles[1], images[1])

    def clear_rectangle_on_other_image():
        redraw_images()

    def redraw_images():
        for img_index in range(2):
            images[img_index] = resize_image(imageio.imread(image_paths[img_index]), scale)
            images[img_index] = cv2.cvtColor(images[img_index], cv2.COLOR_GRAY2BGR)
            for i, (x, y) in enumerate(keypoints[img_index]):
                cv2.circle(images[img_index], (x, y), 5, (0, 255, 0), -1)
                cv2.putText(images[img_index], f"{i + 1}", (x, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(titles[img_index], images[img_index])

    # Display images and set callbacks
    for i, img in enumerate(images):
        print(f"Displaying window with title: {titles[i]}")
        cv2.imshow(titles[i], img)
        cv2.setMouseCallback(titles[i], select_point, param={"img_index": i})

    print("Select keypoints on the reference and moving images.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return keypoints[0], keypoints[1]




def save_to_excel(keypoints1, keypoints2, output_path):
    data = []
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(keypoints1, keypoints2)):
        data.append({
            "Match Index": i + 1,
            "x1": x1/scale,
            "y1": y1/scale,
            "x2": x2/scale,
            "y2": y2/scale,
 
        })
    
    df = pd.DataFrame(data)
    df.to_excel(output_path, index=False)

import logging

def main(image_paths=None, output_dir=None):
    logging.info("Starting manual keypoint selection...")

    # 检查输入路径
    if image_paths is None:
        logging.info("Prompting user for image paths.")
        print("Enter the full paths to the images(reference first and then moving image), separated by a comma:")
        image_paths = input().strip().split(',')
        image_paths = [path.strip() for path in image_paths]

    if output_dir is None:
        logging.info("Prompting user for output directory.")
        print("Enter the output directory path:")
        output_dir = input().strip()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    try:
        # 加载和处理图像
        logging.info(f"Loading images from paths: {image_paths}")
        images = load_images(image_paths)
        logging.info("Images loaded successfully.")

        # 选择关键点
        logging.info("Starting manual keypoint selection process.")
        keypoints_ref, keypoints_mov = manual_select_keypoints(images, titles=["Reference Image", "Moving Image"])
        logging.info(f"Manual keypoints selected: {len(keypoints_ref)} points in reference, {len(keypoints_mov)} points in moving.")

        if len(keypoints_ref) != len(keypoints_mov):
            logging.error("Mismatch in the number of keypoints selected.")
            return

        # 保存关键点到 Excel
        excel_output_path = os.path.join(output_dir, "matched_points.xlsx")
        save_to_excel(keypoints_ref, keypoints_mov, excel_output_path)
        logging.info(f"Matched points saved to {excel_output_path}")

    except Exception as e:
        logging.error(f"Error during manual keypoint selection: {str(e)}", exc_info=True)


    images = load_images(image_paths)
    reference_image = cv2.convertScaleAbs(images[0])
    moving_image = cv2.convertScaleAbs(images[1])

    blurred_reference = gaussian(reference_image, sigma=2)
    blurred_moving = gaussian(moving_image, sigma=2)

    threshold_scale = 1.3  # Increase threshold by 30%

    adjusted_binary_ref = adjust_threshold(blurred_reference, threshold_scale)
    adjusted_binary_ref = remove_small_objects(adjusted_binary_ref, min_size=1000)
    adjusted_binary_ref = binary_closing(adjusted_binary_ref, selem=disk(5))

    adjusted_binary_mov = adjust_threshold(blurred_moving, threshold_scale)
    adjusted_binary_mov = remove_small_objects(adjusted_binary_mov, min_size=1000)
    adjusted_binary_mov = binary_closing(adjusted_binary_mov, selem=(disk(5)))

    masked_reference_image = apply_mask(reference_image, adjusted_binary_ref.copy())
    masked_moving_image = apply_mask(moving_image, adjusted_binary_mov.copy())

    enhanced_reference_image = enhance_contrast(masked_reference_image)
    enhanced_moving_image = enhance_contrast(masked_moving_image)

    ref_image_for_selection = cv2.cvtColor(reference_image, cv2.COLOR_GRAY2BGR)
    mov_image_for_selection = cv2.cvtColor(moving_image, cv2.COLOR_GRAY2BGR)

    # 创建肿瘤形状掩码和计算质心
    circle_mask, masked_reference_image_circle = create_circle_mask(reference_image)
    circle_mask, masked_moving_image_circle = create_circle_mask(moving_image)
    largest_contour_mask_ref = create_largest_contour_mask(masked_reference_image_circle)
    largest_contour_mask_mov = create_largest_contour_mask(masked_moving_image_circle)
    center_ref = compute_centroid(largest_contour_mask_ref)
    center_mov = compute_centroid(largest_contour_mask_mov)

   
    # 手动选择分区点


    """use the second part of code to get the points"""
    images=[imageio.imread(path) for path in image_paths]
    print("Select keypoints on the reference and moving images")
    keypoints_ref, keypoints_mov= manual_select_keypoints(
        [images[0], images[1]], center_ref, titles=["Reference Image", "Moving Image"],scale=0.5)
    print('until here')
    if len(keypoints_ref) != len(keypoints_mov):
        print("Error: The number of keypoints selected on both images must be the same.")
        return

    
    excel_output_path = os.path.join(output_dir)
    save_to_excel(keypoints_ref, keypoints_mov, excel_output_path)
    print(f"Matched points and distances saved to {excel_output_path}")



    #imageio.imwrite(os.path.join(output_dir, 'reference_image_with_manual_kp.tif'), ref_image_for_selection)
    #imageio.imwrite(os.path.join(output_dir, 'moving_image_with_manual_kp.tif'), mov_image_for_selection)

    print("Images with manually selected keypoints saved.")

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(cv2.cvtColor(ref_image_for_selection, cv2.COLOR_BGR2RGB))
    axs[0].set_title('Reference Image with Keypoints')
    axs[1].imshow(cv2.cvtColor(mov_image_for_selection, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Moving Image with Keypoints')
    plt.show()

if __name__ == "__main__":
    if 'image_paths' not in globals():
        print("Enter the full paths to the images(reference first and then moving image), separated by a comma:")
        image_paths = input().strip().split(',')
        image_paths = [path.strip() for path in image_paths]  # Remove extra spaces
    '''image_paths = [
        r'\\eng-fs1.win.rpi.edu\Mills-Lab\Researcher Data\Subbir Parvej\Images\HCT\Embedded MCTS\14th\T1\manual_align_lineROI\\168.tif',
        r'\\eng-fs1.win.rpi.edu\Mills-Lab\Researcher Data\Subbir Parvej\Images\HCT\Embedded MCTS\14th\T1\manual_align_lineROI\\248.tif'
    ]'''
    
    if 'output_dir' not in globals():
        print("Enter the output directory path:")
        output_dir = input().strip()
        
    main(image_paths, output_dir)
    
    