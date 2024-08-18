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
from scipy.stats import kurtosis

GREEN = '\033[92m'
ORANGE = '\033[38;5;208m'
RED = '\033[91m'
RESET = '\033[0m'

def main(preprocessing, loading, processing, presenting, analyzing):
    '''
    Parameters
    '''
    # folders
    camera = './assets/olympus_tg6_parameters.csv'
    img_path='./assets/examples'
    depth_output='./depth_vis'
    segments_output = './segment_vis'
    Zack_img = './assets/transformed_TMFM0126.JPG'
    rescape_output = './rescape_vis'
    maximum_depth = 500 # maximum relative distance from camera to the closest obj in 3D model 8192
    # 0 - 254 object segment ID, 255 background ID, 256 full image
    segment_IDs = [256]  # 3
    Target_compression_factor = 0.315
    # the output from ReScape 
    # {TMFM0126.JPG, 0.315, 1088}, 
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
    if preprocessing:
        # Generate image's depth map file
        deepit()
        # Generate image's segment mask
        segit()
        
        deepit(img_path='./rescape_vis', outdir='./rescape_vis', encoder= 'vitl')

    '''
    loading
    '''
    if loading:
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
            # Convert the images to NumPy arrays
            color_np = np.asarray(color_image, dtype=np.uint8)
            depth_map_array  = np.loadtxt(depth_file[0])
            depth_np = np.asarray(depth_map_array, dtype=np.uint16)

            # # Plot the depth map
            # plt.imshow(depth_np, cmap='viridis', interpolation='none')
            # plt.show()

            # inverse the depth value
            depth_np = np.max(depth_np) - depth_np

            # distance regression
            return_result = depth_bruteforce(color_np, depth_np, maximum_depth, color_image, intr_coeffs)
            return_result = np.array(return_result)

            '''
            presenting
            '''
            if presenting:
                present_bf_result(return_result, Target_compression_factor, imgID)
                # best_depth = return_result[np.argmax(return_result[:, 1]), 0] # best_depth = 180
                best_depth = return_result[find_closest_index(return_result, Target_compression_factor), 0]

                depth_np = np.asarray(depth_np + best_depth, dtype=np.uint16)

                # Masking
                # color_np, depth_np = masking(color_np, depth_np, seg_folder[0], segment_IDs)    

                pcd = create_pcd(depth_np, color_np, intr_coeffs, maximum_depth)
                present_pcd(pcd)

                # pcd_rotated_transformed, rotation_matrix = rotate_camera(pcd)

                # compare_topdownview(pcd, color_image, Zack_img)
                
                # # create top down view
                # combined_array = inverse_project(depth_np, color_np, rotation_matrix)
                # project_top_view(combined_array)

                # Load a point cloud from a PLY file

                present_pcd_series()
    
    '''
    analyzing
    '''
    if analyzing:

        inputfiles = os.listdir(rescape_output)
        imagefiles = [filename for filename in inputfiles if filename.endswith('.jpg')]
        imagefiles.sort()
        mapfiles = [filename for filename in inputfiles if filename.endswith('.txt')]
        maskfile = [filename for filename in inputfiles if filename.endswith('mask.csv')]
        cpfile = [filename for filename in inputfiles if filename.endswith('compression_factors.csv')]
        rafile = [filename for filename in inputfiles if filename.endswith('rotation_angles.csv')]
        
        # for mapfile in mapfiles:
        #     depth_map_array  = np.uint8(np.loadtxt(os.path.join(rescape_output, mapfile)))
        #     depth_map = cv2.applyColorMap(depth_map_array, cv2.COLORMAP_INFERNO)
        #     depth_map_grey = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY) 
        #     cv2.imwrite((mapfile[:mapfile.rfind('.')] + '_color.png'), depth_map)
        #     cv2.imwrite((mapfile[:mapfile.rfind('.')] + '_grey.png'), depth_map_grey)

        masks = pd.read_csv(os.path.join(rescape_output, maskfile[0]), header=None).to_numpy()
        compression_factors = pd.read_csv(os.path.join(rescape_output, cpfile[0]), header=None).to_numpy()
        rotation_angles = pd.read_csv(os.path.join(rescape_output, rafile[0]), header=None).to_numpy()
        counter = 0
        kurtosis_vals = []
        maps = []
        iteration = []
        cf = []
        ra = []
        # down sample
        for mapfile, mask, compression_factor, rotation_angle in zip(mapfiles, masks, compression_factors, rotation_angles):
            counter += 1
            if counter % 1 == 0:
                depth_map = np.uint8(np.loadtxt(os.path.join(rescape_output, mapfile)))
                trimmed_depth_map = trim_map(depth_map, mask)

                kurtosis_val = kurtosis(trimmed_depth_map)

                kurtosis_vals.append(kurtosis_val)
                # maps.append(trimmed_depth_map_1d)
                maps.append(trimmed_depth_map)

                iteration.append(counter)
                cf.append(compression_factor[0])
                ra.append(rotation_angle[0])

        present_kurtosis(kurtosis_vals, maps, iteration, cf, ra)

if __name__ == '__main__':
    preprocessing = False
    loading= True
    processing = True
    presenting = True
    analyzing = False
    main(preprocessing, loading, processing, presenting, analyzing)