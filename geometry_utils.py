import open3d as o3d
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from sklearn.decomposition import TruncatedSVD
from vedo import Plane, Arrow
import math
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from PIL import Image
from tqdm import tqdm
import os
from scipy.spatial import cKDTree
from vedo import Plotter, Points
from datetime import datetime
from scipy.stats import norm
import itertools
import scipy.spatial as spatial
from itertools import combinations
import seaborn as sns
# import matplotlib.cm as cm

GREEN = '\033[92m'
ORANGE = '\033[38;5;208m'
RED = '\033[91m'
RESET = '\033[0m'

def depth_bruteforce(color_np, depth_np, maximum_depth, step_size, intr_coeffs):
    """
    Brute force the distance between the image plane and 3D point cloud
    
    Parameters:
    color_np: image color array.
    depth_np: image depth array.
    maximum_depth: depth iteration upper limit.
    step_size: depth iteration step size.
    intr_coeffs: camera parameters.
    
    Returns:
    brute force search result [[compression_factor, depth],...]
    """

    print(f"\n{GREEN} Depth distance brute force start. {RESET}")

    return_result_01 =[]
    return_result_02 =[]

    # Initialize tqdm with total number of steps
    with tqdm(total=maximum_depth // step_size, desc="cranking ...") as pbar:
        for  dist_imgplane2closeobj in range(0, maximum_depth, step_size):
            # convert the relative distant from point cloud to image plane to relative distance to optical center
            depth_np_local_variable = np.asarray(depth_np + dist_imgplane2closeobj, dtype=np.uint16)
            # Create Open3D Image objects
            pcd = create_pcd(depth_np_local_variable, color_np, intr_coeffs, maximum_depth)
            # save the model
            filename = "point_cloud_" + str(dist_imgplane2closeobj).zfill(3) + ".ply"

            present_pcd(pcd, filename)
            # rotate the camera to top of the pcd
            pcd_rotated_transformed, _ = rotate_camera(pcd)
            # calculate compressor factor 
            cp_value = calculate_compressor_factor(pcd_rotated_transformed)
            # depth analysis
            depth_range = depth_analysis(pcd_rotated_transformed)

            # attach to return result
            return_result_01.append((dist_imgplane2closeobj, cp_value))
            return_result_02.append(depth_range)
            # print(f"\nDistance to image plane: {dist_imgplane2closeobj} ")
            pbar.update(1)

    print(f"\n{GREEN} Depth distance brute force completes. {RESET}")

    return return_result_01, return_result_02

def rotate_camera(pcd):
    """
    rotate the camera to the top of point cloud
    
    Parameters:
    pcd: point cloud
    
    Returns:
    rotated point cloud and rotation matrix
    """
    points = np.asarray(pcd.points)

    normal_fit, centroid = fit_plane(points)
    rotation_matrix = calculate_rotation_matrix(normal_fit)
    # Rotation is applied by matrix multiplication
    rotated_points = np.dot(points, rotation_matrix.T)  
    # Transformation
    rotated_points = rotated_points + centroid 
    # Update the point cloud with the rotated points
    pcd.points = o3d.utility.Vector3dVector(rotated_points)

    # debug    
    # R_normal_fit, R_centroid = fit_plane(rotated_points)
    # Create a line set for the normal vector
    # normal_end = R_centroid + R_normal_fit/10
    # lines = [[0, 1]]
    # line_set = o3d.geometry.LineSet(
    #     points=o3d.utility.Vector3dVector([R_centroid, normal_end]),
    #     lines=o3d.utility.Vector2iVector(lines)
    # )
    # line_set.paint_uniform_color([1, 0, 0])
    # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0, 0, 0])
    # axes.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # o3d.visualization.draw_geometries([pcd, line_set, axes])
    # debug end

    return pcd, rotation_matrix

def create_pcd(depth_np, color_np, intr_coeffs, maximum_depth):
    """
    Create point cloud with depth map array, image pixel colors array, camera parameters and depth limit.
    
    Parameters:
    depth_np: depth map array
    color_np: image pixel color array
    intr_coeffs: camera intrinsic matrix  parameters
    maximum_depth: depth limit
    
    Returns:
    point cloud
    """    

    # Create Open3D Image objects
    color_o3d = o3d.geometry.Image(color_np)
    depth_o3d = o3d.geometry.Image(depth_np)

    # debug       
    # vis = o3d.visualization.Visualizer()
    # vis.create_window("Depth Map Visualization", width=4000, height=3000)
    # # Add depth image to the visualizer
    # vis.add_geometry(depth_o3d)
    # # Run the visualizer
    # vis.run()
    # # Destroy the window after the visualization
    # vis.destroy_window()
    # debug end  

    # Create an RGBDImage object
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale = 1, depth_trunc= maximum_depth + 255, convert_rgb_to_intensity=False)
    height, width = color_np.shape[:2]
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, intr_coeffs[0][0], intr_coeffs[1][1], intr_coeffs[0][2], intr_coeffs[1][2])
    # Default extrinsic matrix
    extrinsic = np.eye(4)

    # Create point cloud from RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image, intrinsic, extrinsic)
    # flip pcd
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd

def find_closest_index(array, target):
    """
    Find the index of the closest value to the target in the array.
    
    Parameters:
    array: A NumPy array of values.
    target: The target value to find the closest match for.
    
    Returns:
    The index of the closest value in the array.
    """
    array = np.asarray(array[:,1])  # Ensure input is a NumPy array
    # Compute the absolute difference between each element and the target
    differences = np.abs(array - target)
    # Find the index of the smallest difference
    closest_index = np.argmin(differences)
    return closest_index

def present_bf_result(return_result, Target_compression_factor, imgID):
    """
    Present the brute force search result
        
    Parameters:
    return_result: brute force search result
    Target_compression_factor: image's compression factor, the output of rescape algorithm
    imgID: image ID

    Returns:
    NA
    """        
    # plot the brute force result
    plt.figure(figsize=(10, 6))

    # plot the optimal depth 
    best_fit = return_result[find_closest_index(return_result, Target_compression_factor), 0]

    plt.axvline(best_fit, color='purple', linestyle='--', label='Optimized Relative distance')
    plt.plot(return_result[:, 0], return_result[:, 1], color='g', linestyle='--')
    plt.scatter(return_result[:, 0], return_result[:, 1], color='g', label="Compression Factor")
    plt.annotate(
        f'({best_fit}, {return_result[find_closest_index(return_result, Target_compression_factor), 1]:.3f})', 
        (best_fit, return_result[find_closest_index(return_result, Target_compression_factor), 1]), 
        textcoords="offset points", 
        xytext=(5,5), 
        ha='right'
        )
    plt.legend(loc='best')
    plt.title('Compression Factor vs Relative distance')
    plt.xlabel('Relative distance')
    plt.ylabel('Compression Factor')
    plt.xlim(0, np.max(return_result[:, 0]))
    plt.grid(True)

    # save the plot to the specified directory
    plot_filename = f'brute_force_result_{imgID}'
    plot_save_path = os.path.join(plot_filename)
    plt.savefig(plot_save_path)

    # display the plot
    # plt.show()

    plt.close()

def present_pcd(pcd, filename=""):
    """
    Display or save pcd
    
    Parameters:
    pcd: point cloud
    filename: file to save or display if filename is empty 
    
    Returns:
    NA
    """    
    # Create coordinate axes geometry
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0, 0, 0])
    axes.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    if filename =="":
        # display
        o3d.visualization.draw_geometries([pcd, axes])
    else:
        # write to a file
        o3d.io.write_point_cloud(filename, pcd)

def fit_plane(point_cloud):
    """
    Fit a plane to the point cloud using least squares fitting
    
    Parameters:
    point_cloud: point cloud object.
    
    Returns:
    normal_fit, centroid
    """    
    
    centroid = np.mean(point_cloud, axis=0)
    points_centered = point_cloud - centroid
    points_centered = points_centered

    # Truncated SVD, which computes only the top 𝑘 singular values and corresponding singular vectors. 
    # This significantly reduces the memory and computational requirements.
    # Perform truncated SVD
    # Specify the number of singular values and vectors you want
    n_components = 3  
    svd = TruncatedSVD(n_components=n_components)

    # Optionally convert the data type to save memory
    points_centered = points_centered.astype(np.float32)

    svd.fit(points_centered)
    # Access the results
    u = svd.transform(points_centered)
    s = svd.singular_values_
    vh = svd.components_    
    normal_fit = vh[2, :]

    # Compute a point on the plane
    # centroid

    return normal_fit, centroid

def calculate_rotation_matrix(normal_vector):
    """
    calculate the rotation matrix
    
    Parameters:
    normal_vector: plane's normal vector
    
    Returns:
    rotation matrix
    """    

    # Define the target normal vector (aligned with the w-axis)
    target_normal = np.array([0, 0, 1])
    
    # Convert normal_vector to numpy array and normalize it
    normal_vector = np.array(normal_vector)
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    # Compute the rotation axis as the cross product
    rotation_axis = np.cross(normal_vector, target_normal)
    
    # Compute the rotation angle using the dot product
    cos_theta = np.dot(normal_vector, target_normal)
    sin_theta = np.linalg.norm(rotation_axis)
    
    # Create the rotation matrix using axis-angle representation
    if sin_theta != 0:
        rotation_axis = rotation_axis / sin_theta  # Normalize the rotation axis
        rotation_angle = np.arctan2(sin_theta, cos_theta)
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
    else:
        rotation_matrix = np.eye(3)  # No rotation needed if the vectors are already aligned
    
    return rotation_matrix

def present_pcd_series():
    """
    present the multi point clouds with different depth adjustment
    
    Parameters:
    NA
    
    Returns:
    NA
    """    
    # retrieve the pcd from ply file
    current_folder = os.getcwd()
    filenames = os.listdir(current_folder)
    filenames = [filename for filename in filenames if  filename.endswith('.ply')]
    filenames.sort()
    # present four examples
    fn = int(len(filenames) / 4)    
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])
    axes.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])    

    # one by one
    # for filename in filenames:
    #     # Load a mesh from a PLY file
    #     pcd = o3d.io.read_point_cloud(filename)

    #     # Visualize the mesh
    #     o3d.visualization.draw_geometries([axes, pcd])

    # all in one
    pcds = []
    pcds.append(axes)
    pcds.append(o3d.io.read_point_cloud(filenames[0]))
    pcds.append(o3d.io.read_point_cloud(filenames[fn]))
    pcds.append(o3d.io.read_point_cloud(filenames[fn+fn]))
    pcds.append(o3d.io.read_point_cloud(filenames[-1]))
    o3d.visualization.draw_geometries(pcds)

def trim_map(depth_map, mask):
    """
    trim the depth map using the trapezoid corner coordinates
    
    Parameters:
    depth_map: original depth map array
    mask: trapezoid corner coordinates
    
    Returns:
    1d array of depth map
    """        
    # Define the corner coordinates of the trapezoid (in the format (x, y))
    # Ensure the coordinates are ordered correctly
    # print(mask)
    points = np.array([
        [mask[4], mask[5]],  # Top-left corner
        [mask[0], mask[1]],  # Top-right corner
        [mask[2], mask[3]],  # Bottom-right corner
        [mask[6], mask[7]]   # Bottom-left corner
    ], dtype=np.int32)

    # Create a mask with the same dimensions as the image
    mask = np.zeros(depth_map.shape[:2], dtype=np.uint8)

    # Draw the trapezoid on the mask
    cv2.fillConvexPoly(mask, points, 1)

    # Apply the mask to the image
    filtered_array = depth_map[mask == 1]
    array_1d = filtered_array.flatten()

    # display
    # width, height = 800, 600
    # resized_image = cv2.resize(trimmed_image, (width, height))
    # Save or display the trimmed image
    # cv2.imwrite('trimmed_image.jpg', trimmed_image)
    # cv2.imshow('Image', resized_image)
    # # cv2.imshow('Trimmed Image', trimmed_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return array_1d

def present_kurtosis(kurtosis_vals, maps, iteration, compression_factors, rotation_angles):
    """
    present kurtosis result
    
    Parameters:
    kurtosis_vals:
    maps:
    iteration:
    compression_factors:
    rotation_angles:
    
    Returns:
    
    """        
    max_kurtosis = max(kurtosis_vals)
    max_kurtosis_pos = iteration[kurtosis_vals.index(max_kurtosis)]
    max_angle = max(rotation_angles)
    max_angle_pos = iteration[rotation_angles.index(max_angle)]
    best_cf = compression_factors[iteration.index(max_angle_pos)]
    print(f"compression factors: best value {best_cf:.3f}")

    fig, axs = plt.subplots(1, 2, figsize=(18, 6))  # Creates a 1x3 grid of subplots

    axs[0].plot(iteration[1:], kurtosis_vals[1:], linestyle='-', color='b')
    axs[0].axvline(max_kurtosis_pos, color='purple', linestyle='--', label='min value')
    axs[0].set_title(f"kurtosis: max value {max_kurtosis:.3f} @ iteration {max_kurtosis_pos}")
    axs[0].set_xlabel("iteration")

    # axs[1].plot(iteration[1:], compression_factors[1:], linestyle='-', color='r')
    # axs[1].set_title(f"compression factors: best value {best_cf:.2f}")
    # axs[1].set_xlabel("iteration")    

    axs[1].plot(iteration[1:], rotation_angles[1:], linestyle='-', color='g')
    axs[1].axvline(max_angle_pos, color='purple', linestyle='--', label='min value')    
    axs[1].set_title(f"rotation angles: max value {max_angle:.1f} @ iteration {max_angle_pos}")
    axs[1].set_xlabel("iteration")

    plt.tight_layout()
    # plt.show()
    plt.savefig("iteration_summary.png")
    plt.close()

    # for kurtosis, map, counter in zip(kurtosis_vals, maps, iteration):
    #     # Plot histogram
    #     plt.hist(map, bins='auto', density=True, alpha=0.6, color='green', edgecolor='black')
    #     # plt.hist(map, bins=254, range=(1, 255),density=True, alpha=0.6, color='green', edgecolor='black')
    #     # Fit a normal distribution to the data
    #     mu, std = norm.fit(map)
    #     xmin, xmax = plt.xlim()
    #     x = np.linspace(xmin, xmax, 100)
    #     p = norm.pdf(x, mu, std)
    #     plt.plot(x, p, 'k', linewidth=2)
    #     # Add title and labels
    #     plt.title(f'Iteration {counter}: Histogram and Normal Distribution (Kurtosis = {kurtosis:.2f})')
    #     plt.xlabel('Value')
    #     plt.ylabel('Density')

    #     # Show plot
    #     # plt.show()
    #     filename = "Iteration" + str(counter).zfill(3) + "_Kurtosis.png"
    #     plt.savefig(filename)
    #     plt.close()

def calculate_compressor_factor(pcd):
    """
    Parallel project the pcd on z axes to a 2D image, and the calcuate the compressor factor
    
    Parameters:
    pcd: point cloud
    
    Returns:
    compressor factor
    """        
    # Convert the point cloud to a numpy array
    points = np.asarray(pcd.points)
    # Collapse z axes
    xy_points = points[:, :2]
    # xy_points = cv2.normalize(xy_points, None, 0, 255, cv2.NORM_MINMAX)
    # xy_points = xy_points.astype(np.uint8)
    # Compute the convex hull
    hull = spatial.ConvexHull(xy_points)

    # Extract the vertices (points that form the hull)
    hull_points = xy_points[hull.vertices]

    # Function to calculate the area of a quadrilateral given four points
    def quadrilateral_area(pts):
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        x3, y3 = pts[2]
        x4, y4 = pts[3]
        return 0.5 * abs(x1*y2 + x2*y3 + x3*y4 + x4*y1 - y1*x2 - y2*x3 - y3*x4 - y4*x1)

    # Initialize maximum area and the corresponding corners
    max_area = 0
    best_corners = None

    # Check all combinations of four points from the hull vertices
    for combination in combinations(hull_points, 4):
        area = quadrilateral_area(combination)
        if area > max_area:
            max_area = area
            best_corners = combination

    # determine four corners
    centroid = np.mean(hull_points, axis=0)
    top_corners = sorted([corner for corner in best_corners if corner[1] > centroid[1]], key=lambda p: p[1], reverse=True)[:2]
    bottom_corners = sorted([corner for corner in best_corners if corner[1] < centroid[1]], key=lambda p: p[1], reverse=True)[:2]
    # calcuate the top edge and bottom edge length
    if len(top_corners) == 2:
        top_edge = np.linalg.norm(np.array(top_corners[0]) - np.array(top_corners[1]))    
    else:
        top_edge = 1e-10
    if len(bottom_corners) == 2:
        buttom_edge = np.linalg.norm(np.array(bottom_corners[0]) - np.array(bottom_corners[1]))
    else: 
        buttom_edge = 0

    # calculate the compressor factor
    compressor_factor = buttom_edge / top_edge

    # # Display the best corners
    # print("Best Corners with Maximum Area:")
    # for i, corner in enumerate(best_corners, 1):
    #     print(f"Corner {i}: {corner}")

    # print(f"Compressor factor = {compressor_factor:.2f}")

    # # Plotting
    # plt.figure(figsize=(10, 6))

    # # Plot all points
    # plt.scatter(xy_points[:, 0], xy_points[:, 1], s=1, label='Points')

    # # Plot the convex hull
    # plt.plot(hull_points[:, 0], hull_points[:, 1], 'k-', lw=2, label='Convex Hull')

    # # Plot the best corners
    # best_corners_array = np.array(best_corners)
    # plt.scatter(best_corners_array[:, 0], best_corners_array[:, 1], color='red', s=100, label='Best Corners')

    # # Draw lines between the best corners
    # plt.plot(np.append(best_corners_array[:, 0], best_corners_array[0, 0]), 
    #         np.append(best_corners_array[:, 1], best_corners_array[0, 1]), 'r--', lw=2, label='Max Area Quadrilateral')

    # # Labels and legend
    # plt.title('Convex Hull with Maximum Area Quadrilateral')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # poor performance solution, discard.
    # centroid = np.mean(hull_points, axis=0)
    # # Initialize extreme corners
    # top_left = None
    # top_right = None
    # bottom_left = None
    # bottom_right = None

    # # Iterate through the hull vertices to find the extreme corners
    # for point in hull_points:
    #     x, y = point
    #     # Check for top-left corner
    #     if x < centroid[0] and y > centroid[1]:
    #         if top_left is None or (x <= top_left[0] and y >= top_left[1]):
    #             top_left = (x, y)

    #     # Check for top-right corner
    #     if x > centroid[0] and y > centroid[1]:
    #         if top_right is None or (x >= top_right[0] and y >= top_right[1]):
    #             top_right = (x, y)
        
    #     # Check for bottom-left corner
    #     if x < centroid[0] and y < centroid[1]:
    #         if bottom_left is None or (x <= bottom_left[0] and y <= bottom_left[1]):
    #             bottom_left = (x, y)
        
    #     # Check for bottom-right corner
    #     if x > centroid[0] and y < centroid[1]:        
    #         if bottom_right is None or (x >= bottom_right[0] and y <= bottom_right[1]):
    #             bottom_right = (x, y)

    # top_edge = np.linalg.norm(np.array(top_right) - np.array(top_left))
    # bottom_edge = np.linalg.norm(np.array(bottom_right) - np.array(bottom_left))
    # compressor_factor = bottom_edge / top_edge
    # # Display the extreme corners
    # print(f"top left: {top_left}, top right: {top_right}, bottom left: {bottom_left}, bottom right: {bottom_right}")
    # print(f"Compressor factor = {compressor_factor:.2f}")

    # # Plotting
    # plt.figure(figsize=(10, 6))

    # # Plot all points
    # # plt.scatter(xy_points[:, 0], xy_points[:, 1], s=1, label='Points')

    # # Plot the convex hull
    # plt.plot(hull_points[:, 0], hull_points[:, 1], 'k-', lw=2, label='Convex Hull')

    # # Plot the extreme corners
    # plt.scatter(top_left[0], top_left[1], color='red', s=50, label='Extreme Corners - top left')
    # plt.scatter(top_right[0], top_right[1], color='pink', s=50, label='Extreme Corners - top right')
    # plt.scatter(bottom_left[0], bottom_left[1], color='green', s=50, label='Extreme Corners - bottom left')    
    # plt.scatter(bottom_right[0], bottom_right[1], color='blue', s=50, label='Extreme Corners - bottom right')    

    # # Labels and legend
    # plt.title('Convex Hull with Extreme Corners')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return compressor_factor

def topdownview(pcd):
    '''
    arguments: cv2 point cloud, input color image
    collapse 3D object z axis 

    return: 2D image
    '''
    # Option 2, collapse z axes
    # Convert the point cloud to a numpy array
    points = np.asarray(pcd.points)
    # Extract colors (RGB values)
    colors = np.asarray(pcd.colors) * 255  # Convert to 0-255 range
    # Ensure the colors are in the correct format
    colors = colors.astype(np.uint8)

    # Convert the image to a NumPy array
    # color_image_np = np.array(color_image)
    # Get the shape of the image
    # image_shape = colors.shape

    # Image resolution
    # image_resolution = (int(image_shape[0]), int(image_shape[1]))

    xy_points = points[:, :2]

    min_x = np.min(xy_points[:, 0])
    max_x = np.max(xy_points[:, 0])
    min_y = np.min(xy_points[:, 1])
    max_y = np.max(xy_points[:, 1])
    # Compute the image resolution
    width = max_x - min_x
    height = max_y - min_y
    # print(f"WTF! width={width}, height={height}")
    # Adjust for the image coordinate system, ensuring non-negative dimensions
    image_resolution = (int(height), int(width))    

    # Normalize the 2D points to the image resolution
    x_normalized = ((xy_points[:, 0] - np.min(xy_points[:, 0])) / 
                        (np.max(xy_points[:, 0]) - np.min(xy_points[:, 0])) * 
                        (image_resolution[1] - 1)).astype(int)

    y_normalized = ((xy_points[:, 1] - np.min(xy_points[:, 1])) / 
                    (np.max(xy_points[:, 1]) - np.min(xy_points[:, 1])) * 
                    (image_resolution[0] - 1)).astype(int)

    # Create an empty image canvas
    rgb_image = np.zeros((image_resolution[0], image_resolution[1], 3), dtype=np.uint8)

    # Ensure shapes match
    # debug
    assert y_normalized.shape[0] == x_normalized.shape[0] == colors.shape[0], \
        "Mismatch in sizes of y_normalized, x_normalized, and cropped colors_flattened"
    if x_normalized.shape[0] < colors.shape[0]:
        colors = colors[:x_normalized.size]
    # debug end

    # Fill the image canvas with the RGB values
    rgb_image[y_normalized, x_normalized] = colors

    # #  only assign color values to pixels that haven't been assigned yet
    # mask = np.zeros((image_resolution[0], image_resolution[1]), dtype=bool)
    # for i in range(len(x_normalized)):
    #     x = x_normalized[i] 
    #     y = y_normalized[i]
    #     if not mask[y, x]:  # Check if the pixel has not been assigned yet
    #         rgb_image[y, x] = colors[i]
    #         mask[y, x] = True  # Update the mask to indicate that this pixel has been assigned
    #     else:
    #         rgb_image[y, x] = [0,255,0] # fill overlap pixel

    rgb_image = np.flipud(rgb_image)

    # Save image file
    image = Image.fromarray(rgb_image)
    # Get the current date and time
    now = datetime.now()
    # Format the date and time into a string
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    # save the plot to the specified directory
    img_filename = f'2d_{timestamp}.png'
    image.save(img_filename)

    # Plot the image
    # fig, axes = plt.subplots(1, 1, figsize=(12, 6))
    # axes.imshow(rgb_image)
    # axes.set_title('Image')
    # axes.axis('off')  # Hide axes
    # plt.show()

    # Convert NumPy array to an image (OpenCV uses BGR format, so we convert to RGB for display)
    image_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    return image_gray

def collapse_pcd(pcd):
    """
    project 3D point cloud to an 2D image
    
    Parameters:
    pcd: point cloud
    
    Returns:
    2d image
    """            
    present_pcd(pcd)
    # Option 2, collapse z axes
    # Convert the point cloud to a numpy array
    points = np.asarray(pcd.points)
   
    # Step 1: Extract x and y coordinates
    x = points[:, 0]
    y = points[:, 1]

    # Extract colors (RGB values)
    colors = np.asarray(pcd.colors) * 255  # Convert to 0-255 range
    # Ensure the colors are in the correct format
    colors = colors.astype(np.uint8)

    # Step 2: Define image dimensions
    image_width = 600  # Width of the image
    image_height = 600  # Height of the image

    # Step 3: Scale and translate coordinates to fit the image dimensions
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_normalized = (x - x_min) / (x_max - x_min) * (image_width - 1)
    y_normalized = (y - y_min) / (y_max - y_min) * (image_height - 1)

    # Step 4: Create an empty image
    image = np.ones((image_height, image_width, 3), dtype=np.uint8)*255

    # Step 5: Map normalized coordinates to image and set pixel values
    for i in range(len(x_normalized)):
        ix, iy = int(x_normalized[i]), int(y_normalized[i])
        image[iy, ix] = colors[i]  # Set pixel value (e.g., 255 for white)

    image = np.flipud(image)
    # Step 6: Display the image
    plt.imshow(image)
    plt.title('Collapsed Point Cloud Image')
    plt.axis('off')
    plt.show()

def depth_analysis(pcd):
    """
    pcd depth value analysis    
    Parameters:
    pcd: point cloud
    
    Returns:
    relative depth value to normal fit plane
    """            
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    depth = points_centered[:,-1]
    return depth

def present_bf_result_depth_range(depth_range, step_size):
    """
    present the depth value range in each iteration
    Parameters:
    depth_range: depth value array in each iteration
    step_size:
    Returns:
    image file
    """            

    # Create a figure with two subplots: one for the box plot and one for the histogram
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    # Define positions for each boxplot
    positions = list(range(len(depth_range)))
    positions = [a * step_size for a in positions]

    # Define a color map based on the number of datasets
    num_datasets = len(depth_range)
    # Get the colormap and generate color list
    colors = plt.get_cmap('Blues', num_datasets)
    color_list = [colors(i / (num_datasets - 1)) for i in range(num_datasets)]

    # Plotting the box plot
    boxplot_dict = axs[0].boxplot(depth_range,  positions=positions, patch_artist=True, widths=20) # Width of each boxplot

    # Customize each box with colors
    for patch, color in zip(boxplot_dict['boxes'], color_list):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    # Customize whiskers, caps, medians, and fliers
    for whisker in boxplot_dict['whiskers']:
        whisker.set(color='black', linewidth=1.5, linestyle='--')

    for cap in boxplot_dict['caps']:
        cap.set(color='black', linewidth=1.5)


    for flier in boxplot_dict['fliers']:
        flier.set(marker='o', color='black', alpha=0.5)

    # Add grid
    axs[0].grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

    # Adding title and labels
    axs[0].set_title('Box Plot of pcd depth')
    axs[0].set_xlabel('depth map offset')
    axs[0].set_ylabel('pcd depth range')


    # Histogram on the second subplot
    for i, data in enumerate(depth_range):
        # axs[1].hist(data, bins=30, color=color_list[i], edgecolor='black', alpha=0.7, label=f'Dataset {i+1}')
        sns.kdeplot(data, ax=axs[1], bw_adjust=1, linewidth=1, color=color_list[i], alpha=0.7, label=f'depth map offset {i * step_size}')


    axs[1].set_title('Density Plot of pcd depth range')
    axs[1].set_xlabel('pcd depth range')
    axs[1].set_ylabel('frequency')

    # Adding a legend to differentiate datasets
    axs[1].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show plot
    plt.show()

'''
~~~ dead code ~~~
'''
# def PnP(combined_array, img_file, R):
#     world = combined_array[:,0:3]
#     image = cv2.imread(img_file)
#     height, width = int(image.shape[0]), int(image.shape[1])
#     world_points = np.vstack((world[0], 
#                               world[int(height/(R*2)-1)], 
#                               world[int(height/R-1)],
#                               world[int(-height/R)], 
#                               world[int(-height/(R*2))], 
#                               world[-1]), 
#                               dtype=np.float32)
#     print(world_points)
#     image_points = np.array([
#     [0, height],
#     [int(width/2), height],
#     [width, height],
#     [0, 0],
#     [int(width/2), 0],
#     [width, 0],
#     ], dtype=np.float32)
#     print(image_points)

#     # Camera intrinsic parameters (replace with actual values)
#     # These should be obtained from intrinsic calibration of the camera
#     fx = 4444.414403529768
#     fy = 4376.777049965183
#     cx, cy = (width / (2*R), height / (2*R))
#     k1, k2, p1, p2, k3 = (0.39624046444186656,-0.02020324278879381,-0.026154725321408934,-0.06162717412413628,0.21803202103401273)
#     camera_matrix = np.array([[fx, 0, cx],
#                             [0, fy, cy],
#                             [0, 0, 1]], dtype=np.float32)
#     dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)  # Distortion coefficients

#     # Solve for extrinsic parameters
#     ret, rvec, tvec = cv2.solvePnP(world_points, image_points, camera_matrix, dist_coeffs)

#     if ret:
#         # Convert rotation vector to rotation matrix
#         R, _ = cv2.Rodrigues(rvec)
 
#         # tvec[0][0] = tvec[0][0] + 100
#         # print("Rotation Matrix:\n", R)
#         # print("Translation Vector:\n", tvec)

#         # Create the extrinsic matrix
#         extrinsic_matrix = np.hstack((R, tvec))
#         extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))

#         print("Extrinsic Matrix:\n", extrinsic_matrix)
#     else:
#         print("Extrinsic parameters could not be determined.")    
    
#     return extrinsic_matrix

# def cam2world(combined_array, extrinsic_matrix):
#     x, y, z = combined_array.T[0], combined_array.T[1], combined_array.T[2]
#     ones = np.ones(combined_array.shape[0])
#     obj_cam = np.column_stack((x, y, z, ones))

#     # Transform the points using the extrinsic matrix
#     obj_world = (np.linalg.inv(extrinsic_matrix) @ obj_cam.T).T

#     return obj_world

# def depth_optical_center(depth_np, intr_coeffs):

#     dist_imgplane2closeobj = 300 # magic number, reaf use 200
#     depth_np = np.asarray(depth_np, dtype=np.uint16)
#     depth_np = depth_np + dist_imgplane2closeobj
#     depth_np_capped = np.asarray(depth_np, dtype=np.uint16)

#     # # normalize depth array
#     # normalized_depth = depth_np / np.iinfo(depth_np.dtype).max
#     # # focal length of the camera
#     # focal_length_x = intr_coeffs[0, 0]
#     # focal_length_y = intr_coeffs[1, 1]
#     # focal_length = math.sqrt(focal_length_x**2 + focal_length_y**2)
#     # true_depth = normalized_depth * (focal_length + dist_imgplane2closeobj)
#     # # Convert relative depth to absolute depth relative to camera optical center
#     # depth_abs_map = true_depth * normalized_depth
#     # depth_abs_map = np.asarray(depth_abs_map, dtype=np.float32)

#     # return depth_abs_map
#     return depth_np_capped

# Function to convert pixel coordinates to 3D point
# def pixel_to_3d(x, y, depth, intrinsic):
#     fx, fy = intrinsic[0,0], intrinsic[1,1]
#     cx, cy = intrinsic[2,0], intrinsic[2,1]
#     z = depth[y, x]
#     if z == 0:
#         return None
#     x3d = (x - cx) * z / fx
#     y3d = (y - cy) * z / fy
#     return np.array([x3d, y3d, z])

# def accumulated_euclidean_distance(line, depth_np, intr_coeffs):

#     # Find the 3D coordinates of the line pixels
#     line_points = []
#     for (x, y) in line:
#         point = pixel_to_3d(x, y, depth_np, intr_coeffs)
#         if point is not None:
#             line_points.append(point)

#     # Initialize accumulated distance
#     accumulated_distance = 0.0

#     # Calculate accumulated Euclidean distance along the bottom line
#     for i in range(len(line_points) - 1):
#         # Calculate Euclidean distance between consecutive points
#         distance = np.linalg.norm(line_points[i + 1] - line_points[i])
#         accumulated_distance += distance

#     return accumulated_distance

# Relative depth map is typically represents a relative distance from the camera sensor for each each pixel. 
# This relative depth can be converted to an actual distance (in meters) from the camera's optical center 
# (also known as the camera center or the focal point of the camera).
# def add_distance2image_plane(compression_factor, depth_np, intr_coeffs):
#     dist_imgplane2closeobj = 0
#     while dist_imgplane2closeobj <= 256:
#         depth_np = np.asarray(depth_np, dtype=np.uint16)
#         depth_np = depth_np + dist_imgplane2closeobj
#         depth_np_capped = np.asarray(depth_np, dtype=np.uint16)

#         # 2D distance
#         # Define the corner pixel coordinates
#         corner_pixels = [
#             (0, 0),  # Top-left
#             (depth_np_capped.shape[1] - 1, 0),  # Top-right
#             (0, depth_np_capped.shape[0] - 1),  # Bottom-left
#             (depth_np_capped.shape[1]-1, depth_np_capped.shape[0] - 1)  # Bottom-right
#         ]

#         # Find the 3D coordinates of the corner pixels
#         corner_points = []
#         for (x, y) in corner_pixels:
#             point = pixel_to_3d(x, y, depth_np_capped, intr_coeffs)
#             if point is not None:
#                 corner_points.append(point)
#         # Euclidean distance between top edge and bottom edge
#         bottom_edge = np.linalg.norm(corner_points[2][:2] - corner_points[3][:2])
#         top_edge = np.linalg.norm(corner_points[0][:2] - corner_points[1][:2])

#         '''
#         # 3D distance
#         height, width = depth_np_capped.shape
#         # Define the bottom line coordinates as a list of tuples
#         bottom_line = [(x, height - 1) for x in range(width)]
#         # Define the top line coordinates as a list of tuples
#         top_line = [(x, 0) for x in range(width)]
#         bottom_edge = accumulated_euclidean_distance(bottom_line, depth_np_capped, intr_coeffs)
#         top_edge =  accumulated_euclidean_distance(top_line, depth_np_capped, intr_coeffs)
#         '''

#         comp_factor = bottom_edge/top_edge
#         print(f" Bottom_edge={bottom_edge} and top_edge={top_edge}")
#         # Calculate the Euclidean distance using only x and y coordinates

#         if comp_factor >= compression_factor:
#             print(f"\n{GREEN}Target compression factor {compression_factor} \nDistance from closest object to image plane is {dist_imgplane2closeobj} compression factor is {comp_factor:.3f}{RESET}")
#             break
#         else:
#             print(f"\n{GREEN}Target compression factor {compression_factor} \nDistance from closest object to image plane is {dist_imgplane2closeobj} compression factor is {comp_factor:.3f}{RESET}")
#             dist_imgplane2closeobj += 1

#     return np.asarray(depth_np + dist_imgplane2closeobj, dtype=np.uint16)
#     # return depth_np_capped




# def compare_topdownview(pcd, color_image, Zack_img):
#     grayA =  topdownview(pcd, color_image)
    
#     target_image = Image.open(Zack_img).convert('RGB')  # Convert to RGB if needed
#     target_image = np.array(target_image)
#     grayB = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)
#     grayB = cv2.cvtColor(grayB, cv2.COLOR_BGR2GRAY)

#     # Get the dimensions of both images
#     heightA, widthA = grayA.shape
#     heightB, widthB = grayB.shape

#     # Calculate the cropping coordinates
#     crop_y = (heightA - heightB) // 2
#     crop_x = (widthA - widthB) // 2

#     # Ensure the coordinates are within bounds
#     crop_y = max(crop_y, 0)
#     crop_x = max(crop_x, 0)

#     # Crop grayA to the dimensions of grayB
#     grayA_cropped = grayA[crop_y:crop_y + heightB, crop_x:crop_x + widthB]

#     # Plot the images side by side
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#     axes[0].imshow(grayA_cropped, cmap='gray')
#     axes[0].set_title('Image 1')
#     axes[0].axis('off')  # Hide axes

#     axes[1].imshow(grayB, cmap='gray')
#     axes[1].set_title('Image 2')
#     axes[1].axis('off')  # Hide axes

#     # # Display the plot
#     # plt.tight_layout()
#     # plt.show()

#     # save the plot to the specified directory
#     plot_filename = f'topdownview_compare'
#     plot_save_path = os.path.join(plot_filename)
#     plt.savefig(plot_save_path)
#     plt.close()

#     # # Create a visualizer
#     # vis = o3d.visualization.Visualizer()
#     # vis.create_window()

#     # # Add your geometry to the visualizer
#     # vis.add_geometry(pcd)

#     # # Update the visualizer and capture the screenshot
#     # vis.update_geometry(pcd)
#     # vis.poll_events()
#     # vis.update_renderer()

#     # # Capture the screenshot
#     # image = vis.capture_screen_float_buffer(do_render=True)

#     # # Save the image to a file
#     # import matplotlib.pyplot as plt
#     # plt.imsave("rendered_image.png", np.asarray(image))

#     # # Close the visualizer
#     # vis.destroy_window()

# def topdownview(pcd):
#     '''
#     arguments: cv2 point cloud, input color image
#     collapse 3D object z axis 

#     return: 2D image
#     '''
#     # Option 2, collapse z axes
#     # Convert the point cloud to a numpy array
#     points = np.asarray(pcd.points)
#     # Extract colors (RGB values)
#     colors = np.asarray(pcd.colors) * 255  # Convert to 0-255 range
#     # Ensure the colors are in the correct format
#     colors = colors.astype(np.uint8)

#     # # Convert the image to a NumPy array
#     # color_image_np = np.array(color_image)
#     # # Get the shape of the image
#     # image_shape = color_image_np.shape

#     # Image resolution
#     # image_resolution = (int(image_shape[0]), int(image_shape[1]))

#     xy_points = points[:, :2]

#     min_x = np.min(xy_points[:, 0])
#     max_x = np.max(xy_points[:, 0])
#     min_y = np.min(xy_points[:, 1])
#     max_y = np.max(xy_points[:, 1])
#     # Compute the image resolution
#     width = max_x - min_x
#     height = max_y - min_y
#     print(f"WTF! width={width}, height={height}")
#     # Adjust for the image coordinate system, ensuring non-negative dimensions
#     image_resolution = (int(height), int(width))    

#     # Normalize the 2D points to the image resolution
#     x_normalized = ((xy_points[:, 0] - np.min(xy_points[:, 0])) / 
#                         (np.max(xy_points[:, 0]) - np.min(xy_points[:, 0])) * 
#                         (image_resolution[1] - 1)).astype(int)

#     y_normalized = ((xy_points[:, 1] - np.min(xy_points[:, 1])) / 
#                     (np.max(xy_points[:, 1]) - np.min(xy_points[:, 1])) * 
#                     (image_resolution[0] - 1)).astype(int)

#     # Create an empty image canvas
#     rgb_image = np.zeros((image_resolution[0], image_resolution[1], 3), dtype=np.uint8)
#     # rgb_image = np.ones((x_normalized, y_normalized, 3), dtype=np.uint8)*255

#     # Ensure shapes match
#     # debug
#     assert y_normalized.shape[0] == x_normalized.shape[0] == colors.shape[0], \
#         "Mismatch in sizes of y_normalized, x_normalized, and cropped colors_flattened"
#     if x_normalized.shape[0] < colors.shape[0]:
#         colors = colors[:x_normalized.size]
#     # debug end

#     # Fill the image canvas with the RGB values
#     rgb_image[y_normalized, x_normalized] = colors

#     # #  only assign color values to pixels that haven't been assigned yet
#     # mask = np.zeros((image_resolution[0], image_resolution[1]), dtype=bool)
#     # for i in range(len(x_normalized)):
#     #     x = x_normalized[i] 
#     #     y = y_normalized[i]
#     #     if not mask[y, x]:  # Check if the pixel has not been assigned yet
#     #         rgb_image[y, x] = colors[i]
#     #         mask[y, x] = True  # Update the mask to indicate that this pixel has been assigned
#     #     else:
#     #         rgb_image[y, x] = [0,255,0] # fill overlap pixel

#     rgb_image = np.flipud(rgb_image)

#     # Save image file
#     image = Image.fromarray(rgb_image)
#     # Get the current date and time
#     now = datetime.now()
#     # Format the date and time into a string
#     timestamp = now.strftime("%Y%m%d_%H%M%S")
#     # save the plot to the specified directory
#     img_filename = f'2d_{timestamp}.png'
#     image.save(img_filename)

#     # Plot the image
#     # fig, axes = plt.subplots(1, 1, figsize=(12, 6))
#     # axes.imshow(rgb_image)
#     # axes.set_title('Image')
#     # axes.axis('off')  # Hide axes
#     # plt.show()

#     # Convert NumPy array to an image (OpenCV uses BGR format, so we convert to RGB for display)
#     image_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
#     image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

#     return image_gray

# def  simcheck(grayA, grayB, Zack_top_source_y):
    
#     # Get the dimensions of both images
#     heightA, widthA = grayA.shape
#     heightB, widthB = grayB.shape

#     # Calculate the cropping coordinates
#     crop_y = (heightA - heightB) // 2
#     crop_x = (widthA - widthB) // 2

#     # Ensure the coordinates are within bounds
#     crop_y = max(crop_y, 0)
#     crop_x = max(crop_x, 0)

#     # Crop grayA to the dimensions of grayB
#     grayA_cropped = grayA[crop_y:crop_y + heightB, crop_x:crop_x + widthB]
#     # grayA_cropped = grayA[Zack_top_source_y:heightA, crop_x:crop_x + widthB]

#     # # debug
#     # # Display the cropped image
#     # # Example dimensions for the display window
#     # window_width = 800
#     # window_height = 600

#     # # Assuming grayA_cropped is already defined and is a valid grayscale image
#     # # Resize the image to the desired display size
#     # resized_image = cv2.resize(grayA_cropped, (window_width, window_height))

#     # cv2.imshow('Cropped Gray Image A', resized_image)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     # # debug end
    
#     (score, diff) = ssim(grayA_cropped, grayB, full=True)
#     # # Optionally, save the cropped image
#     # print("SSIM: {}".format(score))
#     # cv2.imwrite('./cropped_grayA_{:.5f}.jpg'.format(score), grayA_cropped)
#     # cv2.imwrite('./grayB.jpg', grayB)

#     return score

# def simcheck_v2(grayA, grayB):

#     # Get the dimensions of both images
#     heightA, widthA = grayA.shape
#     heightB, widthB = grayB.shape

#     # Calculate the cropping coordinates
#     crop_y = (heightA - heightB) // 2
#     crop_x = (widthA - widthB) // 2

#     # Ensure the coordinates are within bounds
#     crop_y = max(crop_y, 0)
#     crop_x = max(crop_x, 0)

#     # Crop grayA to the dimensions of grayB
#     grayA_cropped = grayA[crop_y:crop_y + heightB, crop_x:crop_x + widthB]

#     # Compute the absolute difference between the two images
#     difference = cv2.absdiff(grayA_cropped, grayB)

#     # Display the images and the difference
#     plt.figure(figsize=(12, 8))

#     plt.subplot(1, 3, 1)
#     plt.title('Original Image')
#     plt.imshow(grayB)
#     plt.axis('off')

#     plt.subplot(1, 3, 2)
#     plt.title('Distorted Image')
#     plt.imshow(grayA)
#     plt.axis('off')

#     plt.subplot(1, 3, 3)
#     plt.title('Difference')
#     plt.imshow(difference)
#     plt.axis('off')

#     # plt.show()

#     # # Get the current date and time
#     # now = datetime.now()
#     # # Format the date and time into a string
#     # timestamp = now.strftime("%Y%m%d_%H%M%S")
#     # # save the plot to the specified directory
#     # plot_filename = f'topdownview_compare_{timestamp}'
#     # plot_save_path = os.path.join(plot_filename)
#     # plt.savefig(plot_save_path)
#     # plt.close()

# def create_plane_mesh(centroid, normal, size=1.0):
#     """
#     Create a plane mesh given a centroid, normal vector, and size.
#     """
#     # Define the size of the plane
#     half_size = size / 2

#     # Create 4 corners of the plane in a local coordinate system
#     corners_local = np.array([
#         [-half_size, -half_size, 0],
#         [ half_size, -half_size, 0],
#         [ half_size,  half_size, 0],
#         [-half_size,  half_size, 0]
#     ])

#     # Transform the corners to the global coordinate system
#     R = o3d.geometry.get_rotation_matrix_from_xyz(normal)
#     corners_global = (R @ corners_local.T).T + centroid

#     # Create the plane mesh
#     plane_mesh = o3d.geometry.TriangleMesh()
#     plane_mesh.vertices = o3d.utility.Vector3dVector(corners_global)
#     plane_mesh.triangles = o3d.utility.Vector3iVector([
#         [0, 1, 2],
#         [0, 2, 3]
#     ])
#     plane_mesh.compute_vertex_normals()
#     plane_mesh.paint_uniform_color([0.7, 0.7, 0.7])

#     return plane_mesh


# def inverse_project(depth_np, color_np, rotation_matrix):
#     # Extract Pixel Coordinates and Color Values
#     width = depth_np.shape[1]
#     height = depth_np.shape[0]

#     colors = color_np.reshape(-1, 3)
#     colors = downsample_2d_1dim(colors, 10)

#     depth = depth_np.flatten()

#     # Get intrinsic parameters
#     K = intrinsic_from_fov(height, width, 45)  # +- 45 degrees
#     K_inv = np.linalg.inv(K)    

#     # Get pixel coordinates
#     pixel_coords = pixel_coord_np(width, height) 

#     # Apply back-projection: K_inv @ pixels * depth
#     cam_coords = K_inv[:3, :3] @ pixel_coords * depth.flatten()

#     # Apply Camera rotation
#     rotated_cam_coords = np.dot(rotation_matrix, cam_coords)  # Rotation is applied by matrix multiplication
    
#     # downsample
#     rotated_cam_coords = downsample_2d_2dim(rotated_cam_coords, 10)

#     combined_array = np.hstack((rotated_cam_coords.T, colors)) # Array of shape (num_pixels, 6) containing camera coordinates and color values.

#     return combined_array

# def project_top_view(combined_array):
#     """
#     Draw the topview projection
#     """

#     x, y, z = combined_array.T[0], combined_array.T[1], combined_array.T[2]
#     # flip the y-axis to positive upwards
#     y = - y

#     colors = combined_array[:, [3, 4, 5]]/255

#     window_x = (min(x)*1.1, max(x)*1.1)
#     # window_y = (min_longitudinal-3, max_longitudinal)
#     window_y = (min(y)*1.1, max(y)*1.1)


#     # Draw Points
#     fig, axes = plt.subplots(figsize=(12, 12))
#     axes.scatter(x, y, c=colors, s=1) # collapse the z component in the projection matrix, it's world obj
#     axes.set_xlim(window_x)
#     axes.set_ylim(window_y)
#     axes.set_title('Reaf Bird Eye View')
#     plt.axis('on')
#     plt.gca().set_aspect('equal')
#     # Adjust layout
#     # fig.tight_layout()
#     # Save the figure to a file
#     # fig.savefig(f'Birdview_{s}.png')
#     plt.show()

# def downsample_2d_1dim(array, downsampling_factor, allow_trim=True):
#     """
#     Downsample a 2D array along the first dimension by averaging blocks of values.

#     Args:
#     - array (np.ndarray): Input 2D array to be downsampled (shape: (n, 3)).
#     - downsampling_factor (int): Factor by which to downsample the first dimension.
#     - allow_trim (bool): Whether to allow trimming of the array to fit exact downsampling (default True).

#     Returns:
#     - np.ndarray: Downsampled array.
#     """
#     n, m = array.shape
    
#     if m != 3:
#         raise ValueError("The second dimension of the input array must be 3.")
    
#     if not allow_trim and (n % downsampling_factor != 0):
#         raise ValueError(f'Array shape {array.shape} does not evenly divide downsampling factor {downsampling_factor} and allow_trim is False.')
    
#     # Calculate new length after downsampling
#     n_trimmed = (n // downsampling_factor) * downsampling_factor
    
#     # Reshape the array into blocks of the downsampling size
#     reshaped_array = array[:n_trimmed].reshape(-1, downsampling_factor, m)
    
#     # Compute the mean over the blocks along the downsampling axis
#     downsampled_array = reshaped_array.mean(axis=1)
    
#     return downsampled_array

# def pixel_coord_np(width, height):
#     """
#     Pixel in homogenous coordinate
#     Returns:
#         Pixel coordinate:       [3, width * height]
#     """
#     x = np.linspace(0, width - 1, width).astype(int)
#     y = np.linspace(0, height - 1, height).astype(int)
#     [x, y] = np.meshgrid(x, y)
#     return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))

# def intrinsic_from_fov(height, width, fov=90):
#     """
#     Basic Pinhole Camera Model
#     intrinsic params from fov and sensor width and height in pixels
#     Returns:
#     K:      [4, 4]
#     """
#     px, py = (width / 2, height / 2)

#     # Define camera intrinsics
#     # Intrinsic parameters olympus_tg6_parameters.csv
#     fx = 4444.414403529768 # Focal length in x
#     fy = 4376.777049965183 # Focal length in y


#     return np.array([[fx, 0, px, 0.],
#                      [0, fy, py, 0.],
#                      [0, 0, 1., 0.],
#                      [0., 0., 0., 1.]])

# def downsample_2d_2dim(array, downsampling_factor, allow_trim=True):
#     """
#     Downsample a 2D array along the second dimension by averaging blocks of values.

#     Args:
#     - array (np.ndarray): Input 2D array to be downsampled (shape: (3, n)).
#     - downsampling_factor (int): Factor by which to downsample the second dimension.
#     - allow_trim (bool): Whether to allow trimming of the array to fit exact downsampling (default True).

#     Returns:
#     - np.ndarray: Downsampled array.
#     """
#     m, n = array.shape
    
#     if m != 3:
#         raise ValueError("The first dimension of the input array must be 3.")
    
#     if not allow_trim and (n % downsampling_factor != 0):
#         raise ValueError(f'Array shape {array.shape} does not evenly divide downsampling factor {downsampling_factor} and allow_trim is False.')
    
#     # Calculate new length after downsampling
#     n_trimmed = (n // downsampling_factor) * downsampling_factor
    
#     # Reshape the array into blocks of the downsampling size
#     reshaped_array = array[:, :n_trimmed].reshape(m, -1, downsampling_factor)
    
#     # Compute the mean over the blocks along the downsampling axis
#     downsampled_array = reshaped_array.mean(axis=2)
    
#     return downsampled_array

# def project_to_image(pcd, intrinsic):
        
#     # Convert Open3D point cloud to numpy array
#     points = np.asarray(pcd.points)
#     colors = np.asarray(pcd.colors)
#     fx, fy = intrinsic[0][0], intrinsic[1][1] 
#     cx, cy = intrinsic[0][2], intrinsic[1][2] 
#     # Projection
#     x, y, z = points[:, 0], points[:, 1], points[:, 2]
#     # Avoid division by zero
#     # z = np.clip(z, a_min=1e-6, a_max=None)

#     u = (fx * x / z) + cx
#     v = (fy * y / z) + cy

#     # Project points to 2D
#     projected_points = np.stack((u, v), axis=-1)


#     # Define the resolution of the output image
#     # image_width, image_height = intrinsic.width, intrinsic.height
#     image_width, image_height = 4000, 3000
#     image = np.zeros((image_height, image_width))

#     # Normalize and fit the points into the image resolution
#     u, v = projected_points[:, 0], projected_points[:, 1]

#     u = np.clip(np.round(u).astype(int), 0, image_width - 1)
#     v = np.clip(np.round(v).astype(int), 0, image_height - 1)

#     # # Create a scatter plot
#     # plt.figure(figsize=(10, 8))
#     # plt.scatter(u, v, s=1, c=colors, alpha=0.5)  # s is the size of points
#     # plt.title('Projected 2D Points')
#     # plt.xlabel('u (x coordinate)')
#     # plt.ylabel('v (y coordinate)')
#     # plt.xlim(0, np.max(u) + 10)
#     # plt.ylim(0, np.max(v) + 10)
#     # plt.gca().invert_yaxis()  # Invert y axis to match image coordinates
#     # plt.grid(True)
#     # # plt.show()

#     # # Format the date and time into a string
#     # now = datetime.now()
#     # timestamp = now.strftime("%Y%m%d_%H%M%S")
#     # # save the plot to the specified directory
#     # img_filename = f'WD_{timestamp}.png'
#     # plt.savefig(img_filename)
#     # plt.close()

#     # Populate the image array
#     # Use np.bincount for efficient pixel counting
#     indices = np.ravel_multi_index((v, u), (image_height, image_width))
#     histogram = np.bincount(indices, minlength=image_height * image_width)
#     image = histogram.reshape((image_height, image_width))

#     # Debug: Print ranges and counts
#     print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
#     print("Projected points range (u):", np.min(u), np.max(u))
#     print("Projected points range (v):", np.min(v), np.max(v))
#     print("Non-zero pixel count:", np.sum(image > 0))

#     # Normalize the image to 0-255 range
#     # image = np.clip(image, 0, 255)  # Ensure pixel values are within 0-255
#     image = (image / np.max(image) * 255).astype(np.uint8)  # Normalize and convert to uint8

#     # Debug: Print some of the image array values
#     print("Image array values (max, min):", np.max(image), np.min(image))

#     # Display the image
#     plt.imshow(image, cmap='gray', origin='lower')
#     plt.title("2D Projection of Point Cloud")
#     plt.colorbar()  # Add colorbar for better visualization
#     plt.show()

# def perspective_projection(pcd, intr_coeffs):

#     # Extract colors (RGB values)
#     colors = np.asarray(pcd.colors) * 255  # Convert to 0-255 range
#     # Ensure the colors are in the correct format
#     colors = colors.astype(np.uint8)

#     points_3d = np.asarray(pcd.points)

#     # 3x4 perspective projection matrix
#     P = np.hstack([intr_coeffs, [[0],[0],[0]]])

#     # Convert 3D points to homogeneous coordinates (adding a column of ones)
#     points_3d_homogeneous = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])

#     # Project 3D points to 2D using the projection matrix
#     points_2d_homogeneous = P @ points_3d_homogeneous.T

#     # Normalize by the third coordinate to convert from homogeneous to Cartesian coordinates
#     points_2d_homogeneous /= points_2d_homogeneous[2, :]

#     # Extract 2D coordinates
#     points_2d = points_2d_homogeneous[:2, :].T

#     min_x = np.min(points_2d[:, 0])
#     max_x = np.max(points_2d[:, 0])
#     min_y = np.min(points_2d[:, 1])
#     max_y = np.max(points_2d[:, 1])
#     # Compute the image resolution
#     width = max_x - min_x
#     height = max_y - min_y
#     # print(f"WTF! width={width}, height={height}")
#     # Adjust for the image coordinate system, ensuring non-negative dimensions
#     image_resolution = (int(height), int(width))    

#     # Normalize the 2D points to the image resolution
#     x_normalized = ((points_2d[:, 0] - np.min(points_2d[:, 0])) / 
#                         (np.max(points_2d[:, 0]) - np.min(points_2d[:, 0])) * 
#                         (image_resolution[1] - 1)).astype(int)

#     y_normalized = ((points_2d[:, 1] - np.min(points_2d[:, 1])) / 
#                     (np.max(points_2d[:, 1]) - np.min(points_2d[:, 1])) * 
#                     (image_resolution[0] - 1)).astype(int)

#     # Create an empty image canvas
#     rgb_image = np.zeros((image_resolution[0], image_resolution[1], 3), dtype=np.uint8)
#     # rgb_image = np.ones((x_normalized, y_normalized, 3), dtype=np.uint8)*255

#     # Ensure shapes match
#     # debug
#     assert y_normalized.shape[0] == x_normalized.shape[0] == colors.shape[0], \
#         "Mismatch in sizes of y_normalized, x_normalized, and cropped colors_flattened"
#     if x_normalized.shape[0] < colors.shape[0]:
#         colors = colors[:x_normalized.size]
#     # debug end

#     # Fill the image canvas with the RGB values
#     # rgb_image[y_normalized, x_normalized] = colors

#     #  only assign color values to pixels that haven't been assigned yet
#     mask = np.zeros((image_resolution[0], image_resolution[1]), dtype=bool)
#     for i in range(len(x_normalized)):
#         x = x_normalized[i] 
#         y = y_normalized[i]
#         if not mask[y, x]:  # Check if the pixel has not been assigned yet
#             rgb_image[y, x] = colors[i]
#             mask[y, x] = True  # Update the mask to indicate that this pixel has been assigned
#         else:
#             rgb_image[y, x] = [0,255,0] # fill overlap pixel

#     # rgb_image = np.flipud(rgb_image)

#     # Save image file
#     image = Image.fromarray(rgb_image)
#     # Get the current date and time
#     now = datetime.now()
#     # Format the date and time into a string
#     timestamp = now.strftime("%Y%m%d_%H%M%S")
#     # save the plot to the specified directory
#     img_filename = f'2d_{timestamp}.png'
#     image.save(img_filename)