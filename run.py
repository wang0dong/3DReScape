"""
Author: WANG Dong
Date: 20240818
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
from scipy.stats import kurtosis
import fnmatch

GREEN = '\033[92m'
ORANGE = '\033[38;5;208m'
RED = '\033[91m'
RESET = '\033[0m'

def main(preprocessing, preloading, processing, presenting, analyzing):
    '''
    Parameters
    '''
    # folders
    camera = './assets/olympus_tg6_parameters.csv'
    img_path='./assets/examples'
    depth_output='./depth_vis'
    segments_output = './segment_vis'
    rescape_output = './rescape_vis'
    # depth parameters
    maximum_depth = 500 # maximum relative distance from camera to the closest obj in 3D model 8192
    step_size = 100 # step size in brute force search
    # 0 - 254 object segment ID, 255 background ID, 256 full image
    segment_IDs = [256] 
    '''
    pre-processing
    create the depth file and segment masks
    '''
    if preprocessing:
        # Generate image's depth map file
        deepit()
        # deepit(img_path='./rescape_vis', outdir='./rescape_vis', encoder= 'vitl')
        # Generate image's segment mask
        segit()

    '''
    pre-loading
    '''
    if preloading:
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
    if processing:
        for filename in filenames:
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
            # load target compression factor
            rescape_output_folder = rescape_output + '\\' + imgID
            inputfiles = os.listdir(rescape_output_folder)
            pattern = 'compression_factor*.txt'
            matching_file = [file_name for file_name in inputfiles if fnmatch.fnmatch(file_name, pattern)]
            with open(os.path.join(rescape_output_folder, matching_file[0]), 'r') as file:
                content = file.read()
                Target_compression_factor = float(content)

            # Convert the images to NumPy arrays
            color_np = np.asarray(color_image, dtype=np.uint8)
            depth_map_array  = np.loadtxt(depth_file[0])
            depth_np = np.asarray(depth_map_array, dtype=np.uint16)
            # inverse the depth value
            depth_np = np.max(depth_np) - depth_np

            # distance regression
            return_result, depth_range = depth_bruteforce(color_np, depth_np, maximum_depth, step_size, intr_coeffs)
            return_result = np.array(return_result)

            '''
            presenting
            '''
            if presenting:
                # present the previous step brute force search result
                present_bf_result(return_result, Target_compression_factor, imgID)
                # present_bf_result_depth_range(depth_range, step_size)

                # optimized depth value
                best_depth = return_result[find_closest_index(return_result, Target_compression_factor), 0]
                # update the depth mape
                depth_np = np.asarray(depth_np + best_depth, dtype=np.uint16)

                # Masking
                # color_np, depth_np = masking(color_np, depth_np, seg_folder[0], segment_IDs)    

                # create pcd
                pcd = create_pcd(depth_np, color_np, intr_coeffs, maximum_depth)

                # rotate the camera to top of the pcd
                # pcd_rotated_transformed, _ = rotate_camera(pcd)
                # topdownview(pcd_rotated_transformed)
                # collapse_pcd(pcd_rotated_transformed)

                # present the pcd created with optimized depth value
                # present_pcd(pcd)

                # Load a point cloud from a PLY file and present series
                # present_pcd_series()
    
    '''
    analyzing
    '''
    if analyzing:
        # analysis the image depth kurtosis value
        # load the rescape output files of rescape step 6 Brute Force Inverse Perspective Mapping
        _, file_with_extension = os.path.split(filenames[0])
        file_name, _ = os.path.splitext(file_with_extension)
        imgID = file_name
        rescape_output_folder = rescape_output + '\\' + imgID
        inputfiles = os.listdir(rescape_output_folder)
        # load image of each compression iteration 
        imagefiles = [filename for filename in inputfiles if filename.endswith('.jpg')]
        imagefiles.sort()
        # load depth file of each compression iteration
        mapfiles = [filename for filename in inputfiles if filename.endswith('depth.txt')]
        # load destination points of each compression iteration
        maskfile = [filename for filename in inputfiles if filename.endswith('mask.csv')]
        # load compression factor of each compression iteration
        cpfile = [filename for filename in inputfiles if filename.endswith('compression_factors.csv')]
        # load rotation angle of each compression iteration
        rafile = [filename for filename in inputfiles if filename.endswith('rotation_angles.csv')]
        
        # dead code, no need to create the depth color and grey images
        # for mapfile in mapfiles:
        #     depth_map_array  = np.uint8(np.loadtxt(os.path.join(rescape_output, mapfile)))
        #     depth_map = cv2.applyColorMap(depth_map_array, cv2.COLORMAP_INFERNO)
        #     depth_map_grey = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY) 
        #     cv2.imwrite((mapfile[:mapfile.rfind('.')] + '_color.png'), depth_map)
        #     cv2.imwrite((mapfile[:mapfile.rfind('.')] + '_grey.png'), depth_map_grey)

        masks = pd.read_csv(os.path.join(rescape_output_folder, maskfile[0]), header=None).to_numpy()
        compression_factors = pd.read_csv(os.path.join(rescape_output_folder, cpfile[0]), header=None).to_numpy()
        rotation_angles = pd.read_csv(os.path.join(rescape_output_folder, rafile[0]), header=None).to_numpy()

        # initialize
        counter = 0
        kurtosis_vals = []
        maps = []
        iteration = []
        cf = []
        ra = []
        # down sample factor
        downsample_factor = 1

        for mapfile, mask, compression_factor, rotation_angle in zip(mapfiles, masks, compression_factors, rotation_angles):
            counter += 1
            if counter % downsample_factor == 0:
                depth_map = np.uint8(np.loadtxt(os.path.join(rescape_output_folder, mapfile)))
                # trim depth map
                trimmed_depth_map = trim_map(depth_map, mask)
                # calculate kurtosis value
                kurtosis_val = kurtosis(trimmed_depth_map)
                kurtosis_vals.append(kurtosis_val)
                maps.append(trimmed_depth_map)

                iteration.append(counter)
                cf.append(compression_factor[0])
                ra.append(rotation_angle[0])

        present_kurtosis(kurtosis_vals, maps, iteration, cf, ra)

if __name__ == '__main__':
    preprocessing = False
    preloading= True
    processing = True
    presenting = False
    analyzing = True
    main(preprocessing, preloading, processing, presenting, analyzing)