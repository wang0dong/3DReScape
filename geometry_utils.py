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

GREEN = '\033[92m'
ORANGE = '\033[38;5;208m'
RED = '\033[91m'
RESET = '\033[0m'

def topdownview(pcd, color_image):
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
    
    # debug
    # colors = np.asarray(color_image) * 255  # Convert to 0-255 range
    # Flatten the colors array to match the number of points
    # colors_flattened = colors.reshape(-1, 3)  # Flatten to shape (3200*2400, 3)    
    # debug end

    # Ensure the colors are in the correct format
    colors = colors.astype(np.uint8)
    # Convert the image to a NumPy array
    color_image_np = np.array(color_image)

    # Get the shape of the image
    image_shape = color_image_np.shape

    # Image resolution
    image_resolution = (int(image_shape[0]), int(image_shape[1]))

    #################### time very expensive, abandon############################
    # # Process points in chunks to avoid memory issues
    # num_points = np.asarray(points).shape[0]
    # distance_threshold = 0.01  # Adjust the threshold as needed
    # chunk_size = 10000  # Adjust this size based on your system's memory capacity
    # overlapping_indices = []

    # for start in range(0, num_points, chunk_size):
    #     end = min(start + chunk_size, num_points)
    #     chunk = points[start:end, :2]
        
    #     # Use KDTree for efficient nearest-neighbor search within the chunk
    #     tree = cKDTree(chunk)
    #     pairs = tree.query_pairs(distance_threshold)
        
    #     for i, j in pairs:
    #         overlapping_indices.append(start + i)
    #         overlapping_indices.append(start + j)

    # # Convert the list of overlapping indices to a set to remove duplicates
    # overlapping_indices = set(overlapping_indices)
    
    # # Replace overlapping points with NaN
    # points[list(overlapping_indices)] = np.nan
    #################### time very expensive, abandon############################

    xy_points = points[:, :2]

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
    #         rgb_image[y, x] = [255] # fill overlap pixel

    rgb_image = np.flipud(rgb_image)
    # Convert NumPy array to an image (OpenCV uses BGR format, so we convert to RGB for display)
    image_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    return image_gray

def  simcheck(grayA, grayB, Zack_top_source_y):
    
    # Get the dimensions of both images
    heightA, widthA = grayA.shape
    heightB, widthB = grayB.shape

    # Calculate the cropping coordinates
    crop_y = (heightA - heightB) // 2
    crop_x = (widthA - widthB) // 2

    # Ensure the coordinates are within bounds
    crop_y = max(crop_y, 0)
    crop_x = max(crop_x, 0)

    # Crop grayA to the dimensions of grayB
    grayA_cropped = grayA[crop_y:crop_y + heightB, crop_x:crop_x + widthB]
    # grayA_cropped = grayA[Zack_top_source_y:heightA, crop_x:crop_x + widthB]

    # # debug
    # # Display the cropped image
    # # Example dimensions for the display window
    # window_width = 800
    # window_height = 600

    # # Assuming grayA_cropped is already defined and is a valid grayscale image
    # # Resize the image to the desired display size
    # resized_image = cv2.resize(grayA_cropped, (window_width, window_height))

    # cv2.imshow('Cropped Gray Image A', resized_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # debug end
    
    (score, diff) = ssim(grayA_cropped, grayB, full=True)
    # Optionally, save the cropped image
    print("SSIM: {}".format(score))
    cv2.imwrite('./cropped_grayA_{:.5f}.jpg'.format(score), grayA_cropped)
    # cv2.imwrite('./grayB.jpg', grayB)

    return score

def depth_bruteforce(color_np, depth_np, maximum_depth, color_image, target_img, Zack_top_source_y, intr_coeffs):
    print(f"\n{GREEN} Depth distance brute force start. {RESET}")
    step_size = 20
    return_result =[]
    # Initialize tqdm with total number of steps
    with tqdm(total=maximum_depth // step_size, desc="cranking ...") as pbar:
        for  dist_imgplane2closeobj in range(0, maximum_depth, step_size):
            # convert the relative distant from obj to image plane to relative distance to optical center
            # depth_np = add_distance2image_plane(compression_factor, depth_np, intr_coeffs)
            depth_np_local_variable = np.asarray(depth_np + dist_imgplane2closeobj, dtype=np.uint16)
            # Create Open3D Image objects
            pcd = create_pcd(depth_np_local_variable, color_np, intr_coeffs)

            # Create top down view
            grayA  = topdownview(pcd, color_image)

            target_image = Image.open(target_img).convert('RGB')  # Convert to RGB if needed
            target_image = np.array(target_image)
            grayB = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)
            grayB = cv2.cvtColor(grayB, cv2.COLOR_BGR2GRAY)
            score = simcheck(grayA, grayB, Zack_top_source_y)
            return_result.append((dist_imgplane2closeobj, score))
            print(f"\nDistance to image plane: {dist_imgplane2closeobj} ")
            pbar.update(1)

    print(f"\n{GREEN} Depth distance brute force completes. {RESET}")
    return return_result

def create_pcd(depth_np, color_np, intr_coeffs):

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
        color_o3d, depth_o3d, depth_scale = 1000, depth_trunc= 255, convert_rgb_to_intensity=False)
    height, width = color_np.shape[:2]
    
    intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, intr_coeffs[0][0], intr_coeffs[1][1], intr_coeffs[0][2], intr_coeffs[1][2])
    # Define the extrinsic matrix
    extrinsic = np.eye(4)

    # Create point cloud from RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    rgbd_image, intrinsic, extrinsic)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return pcd

def present_bf_result(return_result, imgID):
    # plot the brute force result
    plt.figure(figsize=(10, 6))
    # plot the optimal xcf and print a success message

    best_fit = return_result[np.argmax(return_result[:, 1]), 0]
    plt.axvline(best_fit, color='purple', linestyle='--', label='Optimized Relative distance')
    plt.plot(return_result[:, 0], return_result[:, 1], color='g', label="SSIM")

    plt.annotate(
        f'({best_fit}, {np.max(return_result[:, 1]):.3f})', 
        (best_fit, np.max(return_result[:, 1])), 
        textcoords="offset points", 
        xytext=(5,5), 
        ha='right'
        # arrowprops=dict(facecolor='red', shrink=0.01)
        )
    plt.legend(loc='best')
    plt.title('Structural Similarity Index (SSIM) vs Relative distance')
    plt.xlabel('Relative distance from 0 depth object to image plane')
    plt.ylabel('SSIM (0 - 1)')
    plt.xlim(0, np.max(return_result[:, 0]))
    plt.grid(True)

    # save the plot to the specified directory
    plot_filename = f'brute_force_result_{imgID}'
    plot_save_path = os.path.join(plot_filename)
    plt.savefig(plot_save_path)
    plt.close()

def present_pcd(pcd):
    # Create coordinate axes geometry
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.01, origin=[0, 0, 0])
    axes.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([pcd, axes])

    # def rotate_view(vis):
    #     ctr = vis.get_view_control()
    #     ctr.rotate(10.0, 0.0)
    #     return False

    # o3d.visualization.draw_geometries_with_animation_callback([pcd],
    #                                                           rotate_view)

def compare_topdownview(pcd, color_image, Zack_img):
    grayA =  topdownview(pcd, color_image)
    
    target_image = Image.open(Zack_img).convert('RGB')  # Convert to RGB if needed
    target_image = np.array(target_image)
    grayB = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)
    grayB = cv2.cvtColor(grayB, cv2.COLOR_BGR2GRAY)

    # Get the dimensions of both images
    heightA, widthA = grayA.shape
    heightB, widthB = grayB.shape

    # Calculate the cropping coordinates
    crop_y = (heightA - heightB) // 2
    crop_x = (widthA - widthB) // 2

    # Ensure the coordinates are within bounds
    crop_y = max(crop_y, 0)
    crop_x = max(crop_x, 0)

    # Crop grayA to the dimensions of grayB
    grayA_cropped = grayA[crop_y:crop_y + heightB, crop_x:crop_x + widthB]

    # Plot the images side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(grayA_cropped, cmap='gray')
    axes[0].set_title('Image 1')
    axes[0].axis('off')  # Hide axes

    axes[1].imshow(grayB, cmap='gray')
    axes[1].set_title('Image 2')
    axes[1].axis('off')  # Hide axes

    # # Display the plot
    # plt.tight_layout()
    # plt.show()

    # save the plot to the specified directory
    plot_filename = f'topdownview_compare'
    plot_save_path = os.path.join(plot_filename)
    plt.savefig(plot_save_path)
    plt.close()

    # # Create a visualizer
    # vis = o3d.visualization.Visualizer()
    # vis.create_window()

    # # Add your geometry to the visualizer
    # vis.add_geometry(pcd)

    # # Update the visualizer and capture the screenshot
    # vis.update_geometry(pcd)
    # vis.poll_events()
    # vis.update_renderer()

    # # Capture the screenshot
    # image = vis.capture_screen_float_buffer(do_render=True)

    # # Save the image to a file
    # import matplotlib.pyplot as plt
    # plt.imsave("rendered_image.png", np.asarray(image))

    # # Close the visualizer
    # vis.destroy_window()

'''
~~~ dead code ~~~
'''
# def pixel_coord_np(width, height, mask):
#     """
#     Pixel in homogenous coordinate
#     Returns:
#         Pixel coordinate:       [3, width * height]
#     """
#     x = np.linspace(0, width - 1, width).astype(int)
#     y = np.linspace(0, height - 1, height).astype(int)
#     [x, y] = np.meshgrid(x, y)
#     # return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))

#     x_masked = x[mask]
#     y_masked = y[mask]
#     return np.vstack((x_masked.flatten(), y_masked.flatten(), np.ones_like(x_masked.flatten())))


# def intrinsic_from_fov(height, width, fov=90):
#     """
#     Basic Pinhole Camera Model
#     intrinsic params from fov and sensor width and height in pixels
#     Returns:
#         K:      [4, 4]
#     """
#     px, py = (width / 2, height / 2)
#     # hfov = fov / 360. * 2. * np.pi
#     # fx = width / (2. * np.tan(hfov / 2.))

#     # vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
#     # fy = height / (2. * np.tan(vfov / 2.))

#     # Define camera intrinsics
#     # Intrinsic parameters olympus_tg6_parameters.csv
#     fx = 4444.414403529768 # Focal length in x
#     fy = 4376.777049965183 # Focal length in y
#     # px = 1509.2898034524183 # Principal point x-coordinate (usually the center of the image)
#     # py = 1127.6507038365366 # Principal point y-coordinate (usually the center of the image)


#     return np.array([[fx, 0, px, 0.],
#                      [0, fy, py, 0.],
#                      [0, 0, 1., 0.],
#                      [0., 0., 0., 1.]])

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

# def fit_plane(point_cloud):
#     # Fit a plane to the point cloud using least squares fitting
#     centroid = np.mean(point_cloud, axis=0)
#     points_centered = point_cloud - centroid
#     points_centered = points_centered

#     # # Option 1 full SVD
#     # u, s, vh = np.linalg.svd(points_centered)

#     # Option 2 Truncated SVD, which computes only the top 𝑘 singular values and corresponding singular vectors. 
#     # This significantly reduces the memory and computational requirements.
#     # Perform truncated SVD
#     n_components = 3  # Specify the number of singular values and vectors you want
#     svd = TruncatedSVD(n_components=n_components)
#     svd.fit(points_centered)
#     # Access the results
#     u = svd.transform(points_centered)
#     s = svd.singular_values_
#     vh = svd.components_    

#     normal_fit = vh[2, :]
#     # d_fit = -np.dot(normal_fit, centroid)

#     # Compute a point on the plane
#     point_on_plane = centroid

#     # Create plane object

#     # fitted_plane = Plane(pos=point_on_plane, normal=normal_fit, s=(1, 1), res=(1, 1), c='blue', alpha=0.5)
#     # fitted_plane = Plane(pos=point_on_plane, normal=normal_fit, c='blue', alpha=0.5)
#     fitted_plane = Plane(pos=point_on_plane, normal=normal_fit, s=(250, 200)).c('blue').alpha(0.3)
#     # show(fitted_plane)
#     arrow = Arrow(start_pt=point_on_plane, end_pt=point_on_plane + normal_fit * 10, c='red').legend('Normal Vector')
#     return fitted_plane, arrow, normal_fit, centroid

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

# def project_to_image(pcd, intrinsic):
        
#     # Convert Open3D point cloud to numpy array
#     points = np.asarray(pcd.points)

#     fx, fy = intrinsic.get_focal_length()
#     cx, cy = intrinsic.get_principal_point()
#     # Projection
#     x, y, z = points[:, 0], points[:, 1], points[:, 2]
#     # Avoid division by zero
#     z = np.clip(z, a_min=1e-6, a_max=None)

#     u = (fx * x / z) + cx
#     v = (fy * y / z) + cy

#     # Project points to 2D
#     projected_points = np.stack((u, v), axis=-1)


#     # Define the resolution of the output image
#     image_width, image_height = intrinsic.width, intrinsic.height
#     image = np.zeros((image_height, image_width))

#     # Normalize and fit the points into the image resolution
#     u, v = projected_points[:, 0], projected_points[:, 1]
#     u = np.clip(np.round(u).astype(int), 0, image_width - 1)
#     v = np.clip(np.round(v).astype(int), 0, image_height - 1)


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
#     image = np.clip(image, 0, 255)  # Ensure pixel values are within 0-255
#     image = (image / np.max(image) * 255).astype(np.uint8)  # Normalize and convert to uint8

#     # # Debug: Print some of the image array values
#     # print("Image array values (max, min):", np.max(image), np.min(image))

#     # Display the image
#     plt.imshow(image, cmap='gray', origin='lower')
#     plt.title("2D Projection of Point Cloud")
#     plt.colorbar()  # Add colorbar for better visualization
#     plt.show()

