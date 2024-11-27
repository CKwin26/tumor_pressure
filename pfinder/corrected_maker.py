# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 19:31:44 2024

@author: austa
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import binary_closing, disk, remove_small_objects

# Load and preprocess images
def load_images(paths):
    """
    Load grayscale images from the given file paths.
    """
    images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in paths]
    return images

# Create a mask for the largest contour in the image
def create_largest_contour_mask(image, threshold_scale):
    """
    Create a binary mask for the largest contour in the image.
    """
    blurred_image = gaussian(image, sigma=3)
    otsu_thresh = threshold_otsu(blurred_image)
    binary_mask = blurred_image < otsu_thresh * threshold_scale
    binary_mask = remove_small_objects(binary_mask, min_size=1000)
    binary_mask = binary_closing(binary_mask, footprint=disk(5))
    binary_mask = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary_mask)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    return mask, contours

# Compute the centroid of a binary mask
def compute_centroid(mask):
    """
    Compute the centroid of the largest region in a binary mask.
    """
    moments = cv2.moments(mask)
    c_x = int(moments["m10"] / moments["m00"])
    c_y = int(moments["m01"] / moments["m00"])
    return c_x, c_y

# Generate contour points and their angles relative to the centroid
def cpoints(mask, centroid):
    """
    Generate a list of contour points and their angles relative to the centroid.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    for point in contours[0]:
        x, y = point[0]
        theta = np.arctan2(y - centroid[1], x - centroid[0])
        points.append((x, y, theta))
    return points

# Calculate the best-fit slope for contour points within an angle range
def calculate_best_fit_slope_with_cases(cpoints, angle_in_use, angle_half_range):
    """
    Calculate the best-fit slope of contour points within a given angle range.
    Handles special cases where the angle range wraps around 0 or 360 degrees.
    """
    database_angle = np.array([t for _, _, t in cpoints]) * 180 / np.pi + 180
    database_x = np.array([x for x, _, _ in cpoints])
    database_y = np.array([y for _, y, _ in cpoints])

    if angle_in_use < angle_half_range:  # Case 1: Small angles wrapping around 0
        angle_before = angle_in_use + 360 - angle_half_range
        angle_after = angle_in_use + angle_half_range
        valid_indices1 = np.where((database_angle > angle_before) & (database_angle < 360))[0]
        valid_indices2 = np.where((database_angle > 0) & (database_angle < angle_after))[0]
        valid_x = np.concatenate((database_x[valid_indices1], database_x[valid_indices2]))
        valid_y = np.concatenate((database_y[valid_indices1], database_y[valid_indices2]))
    elif angle_in_use > 360 - angle_half_range:  # Case 2: Large angles wrapping around 360
        angle_before = angle_in_use - angle_half_range
        angle_after = angle_in_use - 360 + angle_half_range
        valid_indices1 = np.where((database_angle > angle_before) & (database_angle < 360))[0]
        valid_indices2 = np.where((database_angle > 0) & (database_angle < angle_after))[0]
        valid_x = np.concatenate((database_x[valid_indices1], database_x[valid_indices2]))
        valid_y = np.concatenate((database_y[valid_indices1], database_y[valid_indices2]))
    else:  # Case 3: Regular angle range
        angle_before = angle_in_use - angle_half_range
        angle_after = angle_in_use + angle_half_range
        valid_indices = np.where((database_angle > angle_before) & (database_angle < angle_after))[0]
        valid_x = database_x[valid_indices]
        valid_y = database_y[valid_indices]

    if len(valid_x) < 2 or len(valid_y) < 2:
        raise ValueError("Not enough points to compute the tangent slope.")

    c = np.polyfit(valid_x, valid_y, 2)

    
    
    return c


def calltansec(inputindex, image, tan_lines, sec_lines, xc, yc, xc2, yc2):
    """
    Visualizes the tangent and secant lines for the given index.
    
    Parameters:
        inputindex: Index of the point for which tangent and secant lines are drawn.
        image: The original image for visualization.
        tan_lines: List of tangent slopes for each point.
        sec_lines: List of secant slopes for each point.
        xc: Centroid x-coordinate.
        yc: Centroid y-coordinate.
        xc2: List of x-coordinates for contour points.
        yc2: List of y-coordinates for contour points.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.scatter(xc2[inputindex], yc2[inputindex], color='red', label='Contour Point')

    # Tangent Line
    x_test = np.linspace(xc2[inputindex] - 250, xc2[inputindex] + 250, 100)
    y_test = tan_lines[inputindex] * x_test - tan_lines[inputindex] * xc2[inputindex] + yc2[inputindex]
    plt.plot(x_test, y_test, color='green', label='Tangent Line')

    # Secant Line
    sign_to_get_sec_x = (-xc + xc2[inputindex]) / abs(xc - xc2[inputindex])
    x_test_sec = np.linspace(xc2[inputindex], xc2[inputindex] + 250 * sign_to_get_sec_x, 100)
    y_test_sec = sec_lines[inputindex] * x_test_sec - sec_lines[inputindex] * xc2[inputindex] + yc2[inputindex]
    plt.plot(x_test_sec, y_test_sec, color='red', label='Secant Line')

    plt.legend()
    plt.title(f"Tangent and Secant Visualization for Point Index {inputindex}")
    plt.show()



# Project moving point to a normal direction
def project_to_refer_normal_with_cases(x1, y1, x2, y2, cpoints, angle_half_range):
    """
    Project the moving point to the normal direction based on the reference point and contour points.
    """
    angle_in_use = np.arctan2(y1 - np.mean([y for _, y, _ in cpoints]),
                              x1 - np.mean([x for x, _, _ in cpoints])) * 180 / np.pi + 180
    coefficients = calculate_best_fit_slope_with_cases(cpoints, angle_in_use, angle_half_range)
    # Tangent slope at point (x, y) using the derivative of the polynomial
    
    tangent_slope = 2 * coefficients[0] * x1 + coefficients[1]

    if tangent_slope == 0:
        normal_slope = float('inf')
    elif tangent_slope == float('inf'):
        normal_slope = 0
    else:
        normal_slope = -1 / tangent_slope

    if normal_slope == float('inf'):
        normal_dx = 0
        normal_dy = 1
    elif normal_slope == 0:
        normal_dx = 1
        normal_dy = 0
    else:
        normal_dx = 1 / (1 + normal_slope**2)**0.5
        normal_dy = normal_slope / (1 + normal_slope**2)**0.5
        
    vector_dx = x2 - x1
    vector_dy = y2 - y1
    projection_length = vector_dx * normal_dx + vector_dy * normal_dy
    corrected_x = x1 + projection_length * normal_dx
    corrected_y = y1 + projection_length * normal_dy

    return corrected_x, corrected_y


def compute_tan_sec_lines(cpoints, angle_half_range):
    """
    Compute tangent and secant slopes for all contour points.

    Parameters:
        cpoints: List of contour points and their angles.
        angle_half_range: Angular range to consider for fitting the tangent line.

    Returns:
        tan_lines: List of tangent slopes for all points.
        sec_lines: List of secant slopes for all points.
    """
    # Compute tangent and secant lines based on 2-degree polynomial fit
    tan_lines = []
    sec_lines = []
    for _, (x, y, angle) in enumerate(cpoints):
        try:
            coefficients = calculate_best_fit_slope_with_cases(cpoints, angle * 180 / np.pi + 180, angle_half_range)
            # Tangent slope at point (x, y) using the derivative of the polynomial
            tangent_slope = 2 * coefficients[0] * x + coefficients[1]
            tan_lines.append(tangent_slope)

            # Secant slope (perpendicular to the tangent)
            if tangent_slope == 0:  # Horizontal tangent line
                secant_slope = float('inf')
            elif tangent_slope == float('inf'):  # Vertical tangent line
                secant_slope = 0
            else:
                secant_slope = -1 / tangent_slope
            sec_lines.append(secant_slope)
        except ValueError as e:
            tan_lines.append(None)
            sec_lines.append(None)
            print(f"Skipping point ({x}, {y}) due to error: {e}")

    return tan_lines, sec_lines
# Main function
import logging

def main(image_path=None, excel_path=None, output_path=None, angle_half_range=None, threshold=None):
    logging.info("Starting corrected maker process...")

    # 检查输入参数
    if image_path is None:
        logging.info("Prompting user for image path.")
        print("Enter the path of the reference image after img_reg:")
        image_path = input().strip()

    if excel_path is None:
        logging.info("Prompting user for Excel path.")
        print("Enter the path of the Excel file to process:")
        excel_path = input().strip()

    if output_path is None:
        logging.info("Prompting user for output path.")
        print("Enter the output path for the processed Excel file:")
        output_path = input().strip()

    

    if threshold is None:
        logging.info("Prompting user for threshold.")
        threshold = float(input("Enter the threshold value (default: 1): ").strip() or 1)
        # Step 1: Load the image and generate the mask
        image = load_images([image_path])[0]
        mask, contours = create_largest_contour_mask(image, float(threshold))
        centroid = compute_centroid(mask)

        
        # Plot the original image with the mask overlay
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.imshow(mask, cmap='Reds', alpha=0.5)  # Overlay mask in red
        plt.scatter(centroid[0], centroid[1], color='blue', label='Centroid', s=10)  # Mark the centroid
        plt.legend()
        plt.title("Image with Mask and Centroid")
        plt.show()

   

    
    # Step 1: Load the image and generate the mask
    image = load_images([image_path])[0]
    mask, contours = create_largest_contour_mask(image,threshold)
    centroid = compute_centroid(mask)
    contour_points = cpoints(mask, centroid)
    
    #show the mask first
    while True:
        
        user_input = input("Do you want this threshold?default=1 (yes/no): ").strip().lower()
        if user_input == "yes":
            print("Skipping this step.")
            break
        elif user_input == "no":
            print("Enter the threshold")
            threshold = input().strip()  # Remove extra space
            # Step 1: Load the image and generate the mask
            image = load_images([image_path])[0]
            mask, contours = create_largest_contour_mask(image, float(threshold))
            centroid = compute_centroid(mask)

            
            # Plot the original image with the mask overlay
            plt.subplot(1, 2, 1)
            plt.imshow(image, cmap='gray')
            plt.imshow(mask, cmap='Reds', alpha=0.5)  # Overlay mask in red
            plt.scatter(centroid[0], centroid[1], color='blue', label='Centroid', s=10)  # Mark the centroid
            plt.legend()
            plt.title("Image with Mask and Centroid")
            plt.show()
            
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
            
    
    if angle_half_range is None:
        logging.info("Prompting user for angle_half_range.")
        angle_half_range = float(input("Enter the angle half range (default: 10): ").strip() or 10)
    
    
    
    
    
    # Step 2: Compute tangent and secant lines for visualization
    tan_lines, sec_lines = compute_tan_sec_lines(contour_points, angle_half_range)
    
    
    while True:
        # Create a new figure for visualization
        plt.figure(figsize=(10, 10))
        plt.imshow(image, cmap='gray')
        plt.scatter(centroid[0], centroid[1], color='blue', label='Centroid', s=50)  # Mark the centroid
    
        # Iterate through all tangent and secant lines
        for selected_index in range(len(tan_lines)):
            # Tangent Line
            x_test = np.linspace(
                contour_points[selected_index][0] - 30,
                contour_points[selected_index][0] + 30,
                100
            )
            y_test = (
                tan_lines[selected_index] * x_test
                - tan_lines[selected_index] * contour_points[selected_index][0]
                + contour_points[selected_index][1]
            )
            plt.plot(x_test, y_test, color='green', alpha=0.6, label='Tangent Line' if selected_index == 0 else "")
    
            
        plt.legend()
        plt.title("Tangent and Secant Lines for All Points")
        plt.show()
    
        # Ask user if they are satisfied with the current angle_half_range
        user_input = input("Do you want this angle_half_range? default=10 (yes/no): ").strip().lower()
        if user_input == "yes":
            print("Skipping this step.")
            break
        elif user_input == "no":
            angle_half_range = input("Enter the new angle_half_range: ").strip()
            
            try:
                angle_half_range = float(angle_half_range)
                tan_lines, sec_lines = compute_tan_sec_lines(contour_points, angle_half_range)
                
               
            except ValueError:
                print("Invalid input. Please enter a numeric value.")
                continue
        else:
            print("Invalid input. Please enter 'yes' or 'no'.")
   
    







    

    # Step 3: Load reference and moving points from the Excel file
    data = pd.read_excel(excel_path)
    x1 = data['x1'].values
    y1 = data['y1'].values
    x2 = data['x2'].values
    y2 = data['y2'].values




    # Step 4: Compute corrected points
    corrected_points = []
    for i in range(len(x1)):
        corrected_x, corrected_y = project_to_refer_normal_with_cases(x1[i], y1[i], x2[i], y2[i], contour_points, angle_half_range)
        r = ((contour_points[i][0]-centroid[0]) **2+ (contour_points[i][1]-centroid[1])**2)**0.5
        corrected_points.append((x1[i], y1[i], x2[i], y2[i], corrected_x, corrected_y,r))
        #corrected_points.append((c_dist, c_disp, r))

    corrected_points_new = []
    for i in range(len(x1)):
        corrected_x, corrected_y = project_to_refer_normal_with_cases(x1[i], y1[i], x2[i], y2[i], contour_points, angle_half_range)
        r = ((contour_points[i][0]-centroid[0]) **2+ (contour_points[i][1]-centroid[1])**2)**0.5
        c_dist = (((x1[i] - centroid[0]) ** 2 + (y1[i] - centroid[1]) ** 2) ** 0.5)/r #distance from centroid to (x1, y1)
        c_disp = (((corrected_x - x1[i]) ** 2 + (corrected_y - y1[i]) ** 2) ** 0.5)/r #displacement from (x1, y1) to corrected (corrected_x, corrected_y)
        #corrected_points.append((x1[i], y1[i], x2[i], y2[i], corrected_x, corrected_y,r))
        corrected_points_new.append((c_dist, c_disp, r))
       



        
    # Step 5: Save corrected points to an Excel file
    #corrected_df = pd.DataFrame(corrected_points, columns=['x1', 'y1', 'x2', 'y2', 'corrected_x', 'corrected_y','r'])
    corrected_df = pd.DataFrame(corrected_points_new, columns=['c_dist', 'c_disp', 'r'])
    corrected_df.to_excel(output_path, index=False)

    # Generate angle and radius information from contour points
    angles = [np.arctan2(y - centroid[1], x - centroid[0]) * 180 / np.pi + 180 for x, y, _ in contour_points]
    radii = [((x - centroid[0]) ** 2 + (y - centroid[1]) ** 2) ** 0.5 for x, y, _ in contour_points]  # Calculate radius from centroid
    
    # Plot Radius vs Angle
    plt.figure(figsize=(10, 6))
    plt.scatter(angles, radii, color='blue', alpha=0.7, label="Radius vs Angle")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Radius (pixels)")
    plt.title("Radius vs Angle")
    plt.grid(alpha=0.5)
    plt.legend()
    
    # Save the Radius vs Angle plot
    radius_vs_angle_path = output_path.replace(".xlsx", "_radius_vs_angle.png")
    plt.savefig(radius_vs_angle_path)
    plt.show()
    
    print(f"Radius vs Angle plot saved at: {radius_vs_angle_path}")


    
    # Step 5: Visualize the results
    plt.figure(figsize=(10, 10))

    # Plot the original image with the mask overlay
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.imshow(mask, cmap='Reds', alpha=0.5)  # Overlay mask in red
    plt.scatter(centroid[0], centroid[1], color='blue', label='Centroid', s=100)  # Mark the centroid
    plt.legend()
    plt.title("Image with Mask and Centroid")

    # Plot the projection results
    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='gray')
    for i in range(len(x1)):
        plt.scatter(x1[i], y1[i], color='blue', label='Reference Point' if i == 0 else "")
        plt.scatter(x2[i], y2[i], color='red', label='Moving Point' if i == 0 else "")
        plt.scatter(corrected_points[i][2], corrected_points[i][3], color='green', label='Corrected Point' if i == 0 else "")
        
        plt.plot([x1[i], corrected_points[i][2]], [y1[i], corrected_points[i][3]], 'k--', label='Projection' if i == 0 else "")
    plt.legend()
    plt.title("Projection of Moving Points onto Normals")

    plt.tight_layout()
    plt.show() 
    # Step 6: Visualize the tangent and secant lines for a selected index
      # Example index
   

for i in range(1):
# Execute the main function
    
    if __name__ == "__main__":
        print(i)
        # the rest of code is to get the none for the excel_path output_path
        image_path=None
        excel_path=None
        output_path=None
        angle_half_range =10 #the angle inside what we fit the curve
        threshold = 1# Remove extra space
        
        
        
        
        
        
        
        
                
        
        
        
        
        main()
        selected_index = 10  # Example index
        
