import argparse
import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default=".\\assets\\TMFM0126.JPG")
    parser.add_argument('--mask_path', type=str, default='.\\outputs\\TMFM0126')    
    args = parser.parse_args()
    # read image and resize
    image = cv2.imread(args.img)
    # width = 691
    # height = 518
    width = 4000
    height = 3000    
    # Resize the image
    # resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    # read meta data
    filenames = os.listdir(args.mask_path)
    filenames = [os.path.join(args.mask_path, filename) for filename in filenames]
    matching_strings = [s for s in filenames if '.csv' in s]
    metadata = pd.read_csv(matching_strings[0])
    # column_names = metadata.columns.tolist()
    # bbox_columns = column_names[2:6]
    # sample test of first mask
    sample_idx = 0
    segmentation_masks = []
    for sample_idx in range(len(metadata)):
        # Draw bounding box on the original image
        # bbox = metadata[metadata['id'] == sample_idx][bbox_columns].values[0].tolist()
        # x_min, y_min, box_width, box_height = bbox
        # x_max, y_max = x_min + box_width, y_min + box_height
        # cv2.rectangle(resized_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green rectangle

        mask_path = [s for s in filenames if ('\\' + str(sample_idx) + '.png') in s][0]
        mask_img = np.array(Image.open(mask_path).convert('L'))
        resized_mask_img = cv2.resize(mask_img, (width, height), interpolation=cv2.INTER_LINEAR)
        # Convert grayscale image to numpy array
        image_array = np.array(resized_mask_img)
        # Thresholding to create a boolean mask (True/False)
        threshold = 128  # Adjust this threshold as needed
        mask = (image_array > threshold)
        segmentation_masks.append(mask)

    overlay_colors = [
    tuple(np.random.randint(0, 256, size=3, dtype=int))  # Generate a random color for each mask
    for _ in range(len(metadata))
    ]

    # overlay = resized_image.copy()
    overlay = image.copy()
    overlay[mask] = (0, 0, 255)  # Red color for segmented region

    # Apply overlays to the original image
    # overlay_image = resized_image.copy()
    overlay_image = image.copy()
    for mask, color in zip(segmentation_masks, overlay_colors):
        overlay_image[mask] = color

    # Combine original image with overlay
    alpha = 0.4  # Transparency factor
    # result = cv2.addWeighted(overlay_image, alpha, resized_image, 1 - alpha, 0)
    result = cv2.addWeighted(overlay_image, alpha, image, 1 - alpha, 0)    

    # Display or save the result
    # cv2.imshow('Segmentation Overlay', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite('result.png', result)
