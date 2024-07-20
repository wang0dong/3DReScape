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

GREEN = '\033[92m'
ORANGE = '\033[38;5;208m'
RED = '\033[91m'
RESET = '\033[0m'

def pixel_coord_np(width, height, mask):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    x = np.linspace(0, width - 1, width).astype(int)
    y = np.linspace(0, height - 1, height).astype(int)
    [x, y] = np.meshgrid(x, y)
    # return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))

    x_masked = x[mask]
    y_masked = y[mask]
    return np.vstack((x_masked.flatten(), y_masked.flatten(), np.ones_like(x_masked.flatten())))


def intrinsic_from_fov(height, width, fov=90):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    Returns:
        K:      [4, 4]
    """
    px, py = (width / 2, height / 2)
    # hfov = fov / 360. * 2. * np.pi
    # fx = width / (2. * np.tan(hfov / 2.))

    # vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
    # fy = height / (2. * np.tan(vfov / 2.))

    # Define camera intrinsics
    # Intrinsic parameters olympus_tg6_parameters.csv
    fx = 4444.414403529768 # Focal length in x
    fy = 4376.777049965183 # Focal length in y
    # px = 1509.2898034524183 # Principal point x-coordinate (usually the center of the image)
    # py = 1127.6507038365366 # Principal point y-coordinate (usually the center of the image)


    return np.array([[fx, 0, px, 0.],
                     [0, fy, py, 0.],
                     [0, 0, 1., 0.],
                     [0., 0., 0., 1.]])

def PnP(combined_array, img_file, R):
    world = combined_array[:,0:3]
    image = cv2.imread(img_file)
    height, width = int(image.shape[0]), int(image.shape[1])
    world_points = np.vstack((world[0], 
                              world[int(height/(R*2)-1)], 
                              world[int(height/R-1)],
                              world[int(-height/R)], 
                              world[int(-height/(R*2))], 
                              world[-1]), 
                              dtype=np.float32)
    print(world_points)
    image_points = np.array([
    [0, height],
    [int(width/2), height],
    [width, height],
    [0, 0],
    [int(width/2), 0],
    [width, 0],
    ], dtype=np.float32)
    print(image_points)

    # Camera intrinsic parameters (replace with actual values)
    # These should be obtained from intrinsic calibration of the camera
    fx = 4444.414403529768
    fy = 4376.777049965183
    cx, cy = (width / (2*R), height / (2*R))
    k1, k2, p1, p2, k3 = (0.39624046444186656,-0.02020324278879381,-0.026154725321408934,-0.06162717412413628,0.21803202103401273)
    camera_matrix = np.array([[fx, 0, cx],
                            [0, fy, cy],
                            [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)  # Distortion coefficients

    # Solve for extrinsic parameters
    ret, rvec, tvec = cv2.solvePnP(world_points, image_points, camera_matrix, dist_coeffs)

    if ret:
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
 
        # tvec[0][0] = tvec[0][0] + 100
        # print("Rotation Matrix:\n", R)
        # print("Translation Vector:\n", tvec)

        # Create the extrinsic matrix
        extrinsic_matrix = np.hstack((R, tvec))
        extrinsic_matrix = np.vstack((extrinsic_matrix, [0, 0, 0, 1]))

        print("Extrinsic Matrix:\n", extrinsic_matrix)
    else:
        print("Extrinsic parameters could not be determined.")    
    
    return extrinsic_matrix

def downsample_2d_1dim(array, downsampling_factor, allow_trim=True):
    """
    Downsample a 2D array along the first dimension by averaging blocks of values.

    Args:
    - array (np.ndarray): Input 2D array to be downsampled (shape: (n, 3)).
    - downsampling_factor (int): Factor by which to downsample the first dimension.
    - allow_trim (bool): Whether to allow trimming of the array to fit exact downsampling (default True).

    Returns:
    - np.ndarray: Downsampled array.
    """
    n, m = array.shape
    
    if m != 3:
        raise ValueError("The second dimension of the input array must be 3.")
    
    if not allow_trim and (n % downsampling_factor != 0):
        raise ValueError(f'Array shape {array.shape} does not evenly divide downsampling factor {downsampling_factor} and allow_trim is False.')
    
    # Calculate new length after downsampling
    n_trimmed = (n // downsampling_factor) * downsampling_factor
    
    # Reshape the array into blocks of the downsampling size
    reshaped_array = array[:n_trimmed].reshape(-1, downsampling_factor, m)
    
    # Compute the mean over the blocks along the downsampling axis
    downsampled_array = reshaped_array.mean(axis=1)
    
    return downsampled_array

def downsample_2d_2dim(array, downsampling_factor, allow_trim=True):
    """
    Downsample a 2D array along the second dimension by averaging blocks of values.

    Args:
    - array (np.ndarray): Input 2D array to be downsampled (shape: (3, n)).
    - downsampling_factor (int): Factor by which to downsample the second dimension.
    - allow_trim (bool): Whether to allow trimming of the array to fit exact downsampling (default True).

    Returns:
    - np.ndarray: Downsampled array.
    """
    m, n = array.shape
    
    if m != 3:
        raise ValueError("The first dimension of the input array must be 3.")
    
    if not allow_trim and (n % downsampling_factor != 0):
        raise ValueError(f'Array shape {array.shape} does not evenly divide downsampling factor {downsampling_factor} and allow_trim is False.')
    
    # Calculate new length after downsampling
    n_trimmed = (n // downsampling_factor) * downsampling_factor
    
    # Reshape the array into blocks of the downsampling size
    reshaped_array = array[:, :n_trimmed].reshape(m, -1, downsampling_factor)
    
    # Compute the mean over the blocks along the downsampling axis
    downsampled_array = reshaped_array.mean(axis=2)
    
    return downsampled_array

def fit_plane(point_cloud):
    # Fit a plane to the point cloud using least squares fitting
    centroid = np.mean(point_cloud, axis=0)
    points_centered = point_cloud - centroid
    points_centered = points_centered

    # # Option 1 full SVD
    # u, s, vh = np.linalg.svd(points_centered)

    # Option 2 Truncated SVD, which computes only the top 𝑘 singular values and corresponding singular vectors. 
    # This significantly reduces the memory and computational requirements.
    # Perform truncated SVD
    n_components = 3  # Specify the number of singular values and vectors you want
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(points_centered)
    # Access the results
    u = svd.transform(points_centered)
    s = svd.singular_values_
    vh = svd.components_    

    normal_fit = vh[2, :]
    # d_fit = -np.dot(normal_fit, centroid)

    # Compute a point on the plane
    point_on_plane = centroid

    # Create plane object

    # fitted_plane = Plane(pos=point_on_plane, normal=normal_fit, s=(1, 1), res=(1, 1), c='blue', alpha=0.5)
    # fitted_plane = Plane(pos=point_on_plane, normal=normal_fit, c='blue', alpha=0.5)
    fitted_plane = Plane(pos=point_on_plane, normal=normal_fit, s=(250, 200)).c('blue').alpha(0.3)
    # show(fitted_plane)
    arrow = Arrow(start_pt=point_on_plane, end_pt=point_on_plane + normal_fit * 10, c='red').legend('Normal Vector')
    return fitted_plane, arrow, normal_fit, centroid

def cam2world(combined_array, extrinsic_matrix):
    x, y, z = combined_array.T[0], combined_array.T[1], combined_array.T[2]
    ones = np.ones(combined_array.shape[0])
    obj_cam = np.column_stack((x, y, z, ones))

    # Transform the points using the extrinsic matrix
    obj_world = (np.linalg.inv(extrinsic_matrix) @ obj_cam.T).T

    return obj_world

def depth_optical_center(depth_np, intr_coeffs):

    dist_imgplane2closeobj = 300 # magic number, reaf use 200
    depth_np = np.asarray(depth_np, dtype=np.uint16)
    depth_np = depth_np + dist_imgplane2closeobj
    depth_np_capped = np.asarray(depth_np, dtype=np.uint16)

    # # normalize depth array
    # normalized_depth = depth_np / np.iinfo(depth_np.dtype).max
    # # focal length of the camera
    # focal_length_x = intr_coeffs[0, 0]
    # focal_length_y = intr_coeffs[1, 1]
    # focal_length = math.sqrt(focal_length_x**2 + focal_length_y**2)
    # true_depth = normalized_depth * (focal_length + dist_imgplane2closeobj)
    # # Convert relative depth to absolute depth relative to camera optical center
    # depth_abs_map = true_depth * normalized_depth
    # depth_abs_map = np.asarray(depth_abs_map, dtype=np.float32)

    # return depth_abs_map
    return depth_np_capped

def topdownview(pcd, color_image):
    '''    
    # Option 1
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # Rotate the point cloud to get a top-down view
    # Rotate 90 degrees around the x-axis
    R = pcd.get_rotation_matrix_from_xyz((np.pi / 2, 0, 0))
    pcd.rotate(R, center=(0, 0, 0))

    # Extract the 2D coordinates (x and y) and RGB colors from the point cloud
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    x = points[:, 0]
    y = points[:, 1]

    # Define the resolution of the 2D image
    image_width = 1000
    image_height = 1000

    # Normalize the coordinates to fit them into an image grid
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()


    x_normalized = ((x - x_min) / (x_max - x_min) * (image_width - 1)).astype(int)
    y_normalized = ((y - y_min) / (y_max - y_min) * (image_height - 1)).astype(int)

    # # Create an empty image grid
    image = np.zeros((1000, 1000, 3), dtype=np.uint8)

    # Paint the pixels
    for i in range(len(x_normalized)):
        if 0 <= x_normalized[i] < image_width and 0 <= y_normalized[i] < image_height:
            image[y_normalized[i], x_normalized[i]] = (colors[i] * 255).astype(np.uint8)

    # Display the image
    plt.imshow(image)
    plt.axis('off')
    # plt.show()

    # Save the 2D image
    plt.imsave("top_down_view.png", image)
    '''
    # Option 2, collapse z axes
    # Convert the point cloud to a numpy array
    points = np.asarray(pcd.points)
    # Extract colors (RGB values)
    colors = np.asarray(pcd.colors) * 255  # Convert to 0-255 range
    # Ensure the colors are in the correct format
    colors = colors.astype(np.uint8)
        # Convert the image to a NumPy array
    color_image_np = np.array(color_image)

    # Get the shape of the image
    image_shape = color_image_np.shape

    # Image resolution
    image_resolution = (int(image_shape[0]), int(image_shape[1]))

    # Set all y-coordinates to zero (collapse along the y-axis)
    points[:, 2] = 0
    
    # Normalize the 2D points to the image resolution
    x_normalized = ((points[:, 0] - np.min(points[:, 0])) / 
                    (np.max(points[:, 0]) - np.min(points[:, 0])) * 
                    (image_resolution[1] - 1)).astype(int)

    y_normalized = ((points[:, 1] - np.min(points[:, 1])) / 
                    (np.max(points[:, 1]) - np.min(points[:, 1])) * 
                    (image_resolution[0] - 1)).astype(int)

    # Create an empty image canvas
    rgb_image = np.zeros((image_resolution[0], image_resolution[1], 3), dtype=np.uint8)

    # Fill the image canvas with the RGB values
    rgb_image[y_normalized, x_normalized] = colors

    rgb_image = np.flipud(rgb_image)
    # Convert NumPy array to an image (OpenCV uses BGR format, so we convert to RGB for display)
    image_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    return image_gray

# Function to convert pixel coordinates to 3D point
def pixel_to_3d(x, y, depth, intrinsic):
    fx, fy = intrinsic[0,0], intrinsic[1,1]
    cx, cy = intrinsic[2,0], intrinsic[2,1]
    z = depth[y, x]
    if z == 0:
        return None
    x3d = (x - cx) * z / fx
    y3d = (y - cy) * z / fy
    return np.array([x3d, y3d, z])

def accumulated_euclidean_distance(line, depth_np, intr_coeffs):

    # Find the 3D coordinates of the line pixels
    line_points = []
    for (x, y) in line:
        point = pixel_to_3d(x, y, depth_np, intr_coeffs)
        if point is not None:
            line_points.append(point)

    # Initialize accumulated distance
    accumulated_distance = 0.0

    # Calculate accumulated Euclidean distance along the bottom line
    for i in range(len(line_points) - 1):
        # Calculate Euclidean distance between consecutive points
        distance = np.linalg.norm(line_points[i + 1] - line_points[i])
        accumulated_distance += distance

    return accumulated_distance

# Relative depth map is typically represents a relative distance from the camera sensor for each each pixel. 
# This relative depth can be converted to an actual distance (in meters) from the camera's optical center 
# (also known as the camera center or the focal point of the camera).
def add_distance2image_plane(compression_factor, depth_np, intr_coeffs):
    dist_imgplane2closeobj = 0
    while dist_imgplane2closeobj <= 256:
        depth_np = np.asarray(depth_np, dtype=np.uint16)
        depth_np = depth_np + dist_imgplane2closeobj
        depth_np_capped = np.asarray(depth_np, dtype=np.uint16)

        # 2D distance
        # Define the corner pixel coordinates
        corner_pixels = [
            (0, 0),  # Top-left
            (depth_np_capped.shape[1] - 1, 0),  # Top-right
            (0, depth_np_capped.shape[0] - 1),  # Bottom-left
            (depth_np_capped.shape[1]-1, depth_np_capped.shape[0] - 1)  # Bottom-right
        ]

        # Find the 3D coordinates of the corner pixels
        corner_points = []
        for (x, y) in corner_pixels:
            point = pixel_to_3d(x, y, depth_np_capped, intr_coeffs)
            if point is not None:
                corner_points.append(point)
        # Euclidean distance between top edge and bottom edge
        bottom_edge = np.linalg.norm(corner_points[2][:2] - corner_points[3][:2])
        top_edge = np.linalg.norm(corner_points[0][:2] - corner_points[1][:2])

        '''
        # 3D distance
        height, width = depth_np_capped.shape
        # Define the bottom line coordinates as a list of tuples
        bottom_line = [(x, height - 1) for x in range(width)]
        # Define the top line coordinates as a list of tuples
        top_line = [(x, 0) for x in range(width)]
        bottom_edge = accumulated_euclidean_distance(bottom_line, depth_np_capped, intr_coeffs)
        top_edge =  accumulated_euclidean_distance(top_line, depth_np_capped, intr_coeffs)
        '''

        comp_factor = bottom_edge/top_edge
        print(f" Bottom_edge={bottom_edge} and top_edge={top_edge}")
        # Calculate the Euclidean distance using only x and y coordinates

        if comp_factor >= compression_factor:
            print(f"\n{GREEN}Target compression factor {compression_factor} \nDistance from closest object to image plane is {dist_imgplane2closeobj} compression factor is {comp_factor:.3f}{RESET}")
            break
        else:
            print(f"\n{GREEN}Target compression factor {compression_factor} \nDistance from closest object to image plane is {dist_imgplane2closeobj} compression factor is {comp_factor:.3f}{RESET}")
            dist_imgplane2closeobj += 1

    return np.asarray(depth_np + dist_imgplane2closeobj, dtype=np.uint16)
    # return depth_np_capped

def project_to_image(pcd, intrinsic):
        
    # Convert Open3D point cloud to numpy array
    points = np.asarray(pcd.points)

    fx, fy = intrinsic.get_focal_length()
    cx, cy = intrinsic.get_principal_point()
    # Projection
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    # Avoid division by zero
    z = np.clip(z, a_min=1e-6, a_max=None)

    u = (fx * x / z) + cx
    v = (fy * y / z) + cy

    # Project points to 2D
    projected_points = np.stack((u, v), axis=-1)


    # Define the resolution of the output image
    image_width, image_height = intrinsic.width, intrinsic.height
    image = np.zeros((image_height, image_width))

    # Normalize and fit the points into the image resolution
    u, v = projected_points[:, 0], projected_points[:, 1]
    u = np.clip(np.round(u).astype(int), 0, image_width - 1)
    v = np.clip(np.round(v).astype(int), 0, image_height - 1)


    # Populate the image array
    # Use np.bincount for efficient pixel counting
    indices = np.ravel_multi_index((v, u), (image_height, image_width))
    histogram = np.bincount(indices, minlength=image_height * image_width)
    image = histogram.reshape((image_height, image_width))

    # Debug: Print ranges and counts
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print("Projected points range (u):", np.min(u), np.max(u))
    print("Projected points range (v):", np.min(v), np.max(v))
    print("Non-zero pixel count:", np.sum(image > 0))

    # Normalize the image to 0-255 range
    image = np.clip(image, 0, 255)  # Ensure pixel values are within 0-255
    image = (image / np.max(image) * 255).astype(np.uint8)  # Normalize and convert to uint8

    # # Debug: Print some of the image array values
    # print("Image array values (max, min):", np.max(image), np.min(image))

    # Display the image
    plt.imshow(image, cmap='gray', origin='lower')
    plt.title("2D Projection of Point Cloud")
    plt.colorbar()  # Add colorbar for better visualization
    plt.show()

def  simcheck(grayA, grayB):
    
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

    # Display the cropped image
    cv2.imshow('Cropped Gray Image A', grayA_cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally, save the cropped image
    cv2.imwrite('path/to/cropped_grayA.jpg', grayA_cropped)

    (score, diff) = ssim(grayA_cropped, grayB, full=True)
    print("SSIM: {}".format(score))


    