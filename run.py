"""
Author: WANG Dong
Date: 20240713
"""
from shelldepthanything import *
from shellsegmentanything import *
from geometry_utils import *
import csv
import os
import re
import numpy as np
import open3d as o3d
import pandas as pd

GREEN = '\033[92m'
ORANGE = '\033[38;5;208m'
RED = '\033[91m'
RESET = '\033[0m'

def main():
    '''
    Parameters
    '''
    # folders
    camera = './assets/olympus_tg6_parameters.csv'
    img_path='./assets/examples'
    depth_output='./depth_vis'
    segments_output = './segment_vis'
    Zack_img = './assets/transformed_TMFM0126.JPG'
    # 0 - 254 object segment ID, 255 background ID, 256 full image
    segment_IDs = [256] 
    compression_factor = 0.295 
    # the output from ReScape 
    # {TMFM0126.JPG, 0.315}, 
    # {PB210074.JPG, 0.295}, 
    # {TMFM0037.JPG, 0.295}, 
    # {P1010150.JPG, 0.25}, 
    # {TMFM0154.JPG, 0.26}, 
    # {TMFM0128.JPG, 0.38},
    # {TMFM0154.JPG, 0.35},

    '''
    pre-processing
    create the depth file and segment masks
    '''
    # Generate image's depth map file
    # deepit()
    # Generate image's segment mask
    # segit()

    '''
    loading
    '''
    # image files
    if os.path.isfile(img_path):
        filenames = [img_path]
    else:
        filenames = os.listdir(img_path)
        filenames = [os.path.join(img_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()
    # depth files
    depfiles = os.listdir(depth_output)
    depfiles =  [os.path.join(depth_output, depthfile) for depthfile in depfiles if not depthfile.startswith('.')]
    depfiles.sort()
    # seg folders
    segfolders = os.listdir(segments_output)
    segfolders =  [os.path.join(segments_output, segfolder) for segfolder in segfolders]
    # load camera parameters
    intr_coeffs = np.array(pd.read_csv(camera, nrows=3, header=None).values.tolist())
    with open(camera, mode='r') as file:
        reader = csv.reader(file)
        # skip to row 4
        for row in range(3):
            next(reader)
        # read the 1x5 distortion coefficients vector
        dist_coeffs = np.array([list(map(float, next(reader)))])

    '''
    processing
    '''
    for filename in tqdm(filenames):
        # load image
        color_image = Image.open(filename).convert('RGB')  # Convert to RGB if needed
        _, file_with_extension = os.path.split(filename)
        file_name, _ = os.path.splitext(file_with_extension)
        imgID = file_name
        # load depth file
        depth_file = [s for s in depfiles if imgID in s]
        if len(depth_file) !=1:
            print(f"{RED} multi depth files are found.{RESET}")    
    # load segment folder
        seg_folder = [s for s in segfolders if imgID in s]

        # Convert the images to NumPy arrays
        color_np = np.asarray(color_image, dtype=np.uint8)
        depth_map_array  = np.loadtxt(depth_file[0])
        depth_np = np.asarray(depth_map_array, dtype=np.uint16)
        # inverse the depth value
        depth_np = np.max(depth_np) - depth_np

        dist_imgplane2closeobj = 0
        while True:

            # convert the relative distant from obj to image plane to relative distance to optical center
            # depth_np = add_distance2image_plane(compression_factor, depth_np, intr_coeffs)
            depth_np = np.asarray(depth_np + dist_imgplane2closeobj, dtype=np.uint16)

            # Masking
            color_np, depth_np = masking(color_np, depth_np, seg_folder[0], segment_IDs)

            #########debug code start###################    
            # depth_np = smooth_depth(depth_np)
            #########debug code end#####################

            # Create Open3D Image objects
            color_o3d = o3d.geometry.Image(color_np)
            depth_o3d = o3d.geometry.Image(depth_np)

            #########debug code start###################    
            # vis = o3d.visualization.Visualizer()
            # vis.create_window("Depth Map Visualization", width=4000, height=3000)

            # # Add depth image to the visualizer
            # vis.add_geometry(depth_o3d)

            # # Run the visualizer
            # vis.run()

            # # Destroy the window after the visualization
            # vis.destroy_window()
            #########debug code end#######################    

            # Create an RGBDImage object
            rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, depth_scale = 1000, depth_trunc=3, convert_rgb_to_intensity=False)
            height, width = color_np.shape[:2]

            intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, intr_coeffs[0][0], intr_coeffs[1][1], intr_coeffs[0][2], intr_coeffs[1][2])

            # Define the extrinsic matrix
            # HOLD
            extrinsic = np.eye(4)

            # Create point cloud from RGBD image
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic, extrinsic)
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # Create coordinate axes geometry
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0, 0, 0])
            axes.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            # o3d.visualization.draw_geometries([pcd, axes])

            # Create top down view
            grayB  = topdownview(pcd, color_image)
            target_image = Image.open(Zack_img).convert('RGB')  # Convert to RGB if needed
            target_image = np.array(target_image)
            grayA = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)
            grayA = cv2.cvtColor(grayA, cv2.COLOR_BGR2GRAY)
            simcheck(grayB, grayA)

            # project_to_image(pcd, intrinsic)
            dist_imgplane2closeobj += 40
            print(dist_imgplane2closeobj)

if __name__ == '__main__':
    main()