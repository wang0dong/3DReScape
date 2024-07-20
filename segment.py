import cv2
import numpy as np
import os
import pandas as pd
from geometry_utils import *
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
# Set backend explicitly
plt.switch_backend('TkAgg')  # Use 'TkAgg', 'Qt5Agg', 'WXAgg', etc. based on your system and preference

import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from scipy.ndimage import median_filter
from geometry_utils import *

def seg(mask_path, width = 4000, height = 3000):
    filenames = os.listdir(mask_path)
    filenames = [os.path.join(mask_path, filename) for filename in filenames]
    matching_strings = [s for s in filenames if '.csv' in s]
    metadata = pd.read_csv(matching_strings[0])

    sample_idx = 0
    # background seg ID 255
    segmentation = np.full((height, width), 255)
    for mask_idx in range(len(metadata)):

        mask_path = [s for s in filenames if ('\\' + str(mask_idx) + '.png') in s][0]
        mask_img = np.array(Image.open(mask_path).convert('L'))
        resized_mask_img = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_LINEAR)
        # Convert grayscale image to numpy array
        image_array = np.array(resized_mask_img)
        # Threshold to create a boolean mask (True/False)
        threshold = 128  # Adjust this threshold as needed
        mask = (image_array > threshold)
        # fill the seg ID 
        segmentation[mask] = mask_idx

    return segmentation


def showme_depth_seg(depth_map_array, mask):

    masked_array = np.ma.masked_where(~mask, depth_map_array)
    x, y = np.meshgrid(np.arange(masked_array.shape[1]), np.arange(masked_array.shape[0]))
    # Plot the 3D surface
    fig = plt.figure(figsize=(10, 8))  # Adjust figure size if needed
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, masked_array, cmap='viridis', edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)  # Add color bar
    # Customize labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Depth')
    plt.title('3D Plot of Masked Depth')
    plt.show()

def smooth_depth(depth_map_array, mask):
    masked_array = np.ma.masked_where(~mask, depth_map_array)
    # masked_array = depth_map_array[mask]
    # Create grid coordinates for 3D plot
    x, y = np.meshgrid(np.arange(masked_array.shape[1]), np.arange(masked_array.shape[0]))

    # Fill masked values using interpolation
    x_flat = x[mask].flatten()
    y_flat = y[mask].flatten()
    z_flat = masked_array[mask].flatten()

    filled_array = griddata((x_flat, y_flat), z_flat, (x, y), method='linear')
    filled_array = np.nan_to_num(filled_array, nan=np.nanmean(z_flat))
    
    # Calculate the gradient to identify dramatic height changes
    grad_x, grad_y = np.gradient(filled_array)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Identify areas with significant increases in height
    # height_increase_mask = (grad_x < 0) | (grad_y < 0)  # Change in either x or y direction

    # Identify areas with significant increases in height using IQR

    # Identify areas with significant increases in height using IQR
    q1 = np.percentile(grad_magnitude, 25)
    q3 = np.percentile(grad_magnitude, 75)
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr

    # threshold = np.mean(grad_magnitude) - 2 * np.std(grad_magnitude)

    # height_increase_mask = height_increase_mask & (grad_magnitude > threshold)
    height_increase_mask = (grad_magnitude > threshold)
    # print(height_increase_mask)

    # Apply median filter to smooth the dramatic height changes
    size = 3  # Size of the filter
    # filtered_array = median_filter(masked_array.filled(np.nan), size=size)  # Using .filled(0) to handle masked values

    depth_array_filtered = masked_array.copy()
    # depth_array_filtered[height_increase_mask] = filtered_array[height_increase_mask]
    # depth_array_filtered[height_increase_mask] = np.nan
    depth_array_filtered[height_increase_mask] = median_filter(masked_array, size=size)[height_increase_mask]

    # # Create figure and 3D axis
    # fig = plt.figure(figsize=(12, 6))  # Adjusted the figure size to be wider for side by side plots

    # # Original plane with dramatic height changes
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax1.set_title('Original Plane with Dramatic Height Changes')
    # surf1 = ax1.plot_surface(x, y, masked_array, cmap='viridis', edgecolor='none')
    # fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # # Modified plane
    # ax2 = fig.add_subplot(122, projection='3d')
    # ax2.set_title('Modified Plane')
    # surf2 = ax2.plot_surface(x, y, depth_array_filtered, cmap='viridis', edgecolor='none')
    # fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    # plt.show()
    depth_array_filtered = depth_array_filtered.filled(np.nan)
    return(depth_array_filtered)

