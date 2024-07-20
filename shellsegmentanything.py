"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Wang Dong
"""
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from scipy.ndimage import median_filter
from geometry_utils import *
import cv2  # type: ignore
from seg_anything.transform import Resize, NormalizeImage, PrepareForNet
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import multiprocessing
import json
import os
from typing import Any, Dict, List
from torchvision.transforms import Compose
import torch.nn.functional as F
import torch

GREEN = '\033[92m'
ORANGE = '\033[38;5;208m'
RED = '\033[91m'
RESET = '\033[0m'

# Set backend explicitly
plt.switch_backend('TkAgg')  # Use 'TkAgg', 'Qt5Agg', 'WXAgg', etc. based on your system and preference


def segit(input='./assets/examples', output='./segment_vis', model_type='vit_h', checkpoint='./assets/checkpoints/sam_vit_h_4b8939.pth'):

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading model...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    _ = sam.to(device=DEVICE)
    output_mode = "binary_mask"
    generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode)

    if not os.path.isdir(input):
        targets = [input]
    else:
        targets = [
            f for f in os.listdir(input) if not os.path.isdir(os.path.join(input, f))
        ]
        targets = [os.path.join(input, f) for f in targets]

    os.makedirs(output, exist_ok=True)

    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            # ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        # NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # PrepareForNet(),
    ])

    # set the number of processes according to the user-submitted percent CPU allocation
    max_processes = multiprocessing.cpu_count()
    num_processes = int(50 * max_processes / 100)

    # ensure at least one logical processor is allocated
    if num_processes < 1:
        num_processes = 1

    # print the submitted processing request
    print(f'\n{GREEN}{"-" * 50}'
            f'\n\nUsing {num_processes} CPU Core computing power...'
            f'\n\n{"-" * 50} {RESET}')

    # use a multiprocessing pool to parallelize the image processing
    with multiprocessing.Pool(processes=num_processes) as pool:
        for t in targets:
            print(f"Processing '{t}'...")
            image = cv2.imread(t)
            if image is None:
                print(f"Could not load '{t}' as an image, skipping...")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            h, w = image.shape[:2]
            image = transform({'image': image})['image']
            masks = generator.generate(image)
            torch.save(masks, 'tensor_data.pt')
        
            # masks = torch.load('./seg_anything/tensor_data.pt')

            base = os.path.basename(t)
            base = os.path.splitext(base)[0]
            save_base = os.path.join(output, base)
            if output_mode == "binary_mask":
                os.makedirs(save_base, exist_ok=False)
                write_masks_to_folder(masks, save_base)
            else:
                save_file = save_base + ".json"
                with open(save_file, "w") as f:
                    json.dump(masks, f)

    print("Done!")

def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return

def upsamplemask(mask_path, width = 4000, height = 3000):
    filenames = os.listdir(mask_path)
    filenames = [os.path.join(mask_path, filename) for filename in filenames]
    matching_strings = [s for s in filenames if '.csv' in s]
    metadata = pd.read_csv(matching_strings[0])

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

def smooth_depth(depth_map_array):

    x, y = np.meshgrid(np.arange(depth_map_array.shape[1]), np.arange(depth_map_array.shape[0]))
    # Fill masked values using interpolation
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = depth_map_array.flatten()

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
    # or 
    # threshold = np.mean(grad_magnitude) - 2 * np.std(grad_magnitude)

    # height_increase_mask = height_increase_mask & (grad_magnitude > threshold)
    height_increase_mask = (grad_magnitude > threshold)

    # Apply median filter to smooth the dramatic height changes
    size = 3  # Size of the filter
    # filtered_array = median_filter(masked_array.filled(np.nan), size=size)  # Using .filled(0) to handle masked values

    depth_array_filtered = masked_array.copy()
    # depth_array_filtered[height_increase_mask] = filtered_array[height_increase_mask]
    # depth_array_filtered[height_increase_mask] = np.nan
    depth_array_filtered[height_increase_mask] = median_filter(masked_array, size=size)[height_increase_mask]

    #########debug code start###################   
    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 6))  # Adjusted the figure size to be wider for side by side plots

    # Original plane with dramatic height changes
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('Original Plane with Dramatic Height Changes')
    surf1 = ax1.plot_surface(x, y, masked_array, cmap='viridis', edgecolor='none')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # Modified plane
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.set_title('Modified Plane')
    surf2 = ax2.plot_surface(x, y, depth_array_filtered, cmap='viridis', edgecolor='none')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    plt.show()
    #########debug code start###################

    depth_array_filtered = depth_array_filtered.filled(np.nan)
    return(depth_array_filtered)

def masking(color_np, depth_np, segments_folder, segment_IDs):

    seg_masks = upsamplemask(segments_folder, width = color_np.shape[1], height = color_np.shape[0])

    big_mask = np.zeros_like(seg_masks)
    for segment_ID in segment_IDs:
        if segment_ID <= 255:
            mask = np.zeros_like(seg_masks)
            mask_bits = (seg_masks == segment_ID)
            mask[mask_bits] = 1
            mask = mask.astype(bool)
            # Logical OR
            big_mask = np.logical_or(big_mask, mask)
        else:
            big_mask = np.ones_like(seg_masks)
            big_mask = big_mask.astype(bool)
            break
    
    # color_np_masked = color_np[mask]
    # depth_np_masked = depth_np[mask]
    mask_expanded = big_mask[:, :, np.newaxis]
    color_np_masked = color_np * mask_expanded
    depth_np_masked = depth_np * big_mask

    return color_np_masked, depth_np_masked