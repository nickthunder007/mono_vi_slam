import numpy as np
import open3d as o3d
import cv2

def depth_to_3d(depth_map, f_x, f_y, c_x, c_y):
   
    # Get the image dimensions
    height, width = depth_map.shape

    # Create a grid of pixel coordinates (x, y)
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten the coordinates and depth map to apply the transformation
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()
    depth_values = depth_map.flatten()

    # Convert depth to 3D points using the back-projection formula
    X = (x_coords - c_x) * depth_values / f_x
    Y = (y_coords - c_y) * depth_values / f_y
    Z = depth_values

    # Stack the 3D points into a single array (X, Y, Z)
    points_3d = np.vstack((X, Y, Z)).T

    return points_3d


def numpy_to_o3d(pcd_array):
    pcd = o3d.geometry.PointCloud()
    
    pcd.points = o3d.utility.Vector3dVector(pcd_array)
    return pcd

def extract_features(image):
    #Detect and compute keypoints and descriptors using ORB.
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features2(des1, des2):
    #Match features using Brute-Force Matcher with Hamming distance.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches



def match_features(kp1, des1, kp2, des2):
    # Match features using Brute-Force Matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    #pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    #pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Compute Homography using RANSAC
    #H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)

    # Filter matches based on RANSAC inliers
    #inlier_matches = [m for m, inlier in zip(matches, mask.ravel()) if inlier]

    return matches


def add_ones(pts):
    #Helper function to add a column of ones to a 2D array (homogeneous coordinates).
    return np.hstack([pts, np.ones((pts.shape[0], 1))])

def triangulate_points(pose1, pose2, pts1, pts2):
    # Initialize the result array to store the homogeneous coordinates of the 3D points
    ret = np.zeros((pts1.shape[0], 4))

    pose1 = np.vstack([pose1, [0, 0, 0, 1]])
    pose2 = np.vstack([pose2, [0, 0, 0, 1]])
 
    # Invert the camera poses to get the projection matrices
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
    delete_idx = []
 
    # Loop through each pair of corresponding points
    for i, p in enumerate(zip(add_ones(pts1), add_ones(pts2))):
        # Initialize the matrix A to hold the linear equations
        A = np.zeros((4, 4))
 
        # Populate the matrix A with the equations derived from the projection matrices and the points
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
 
        # Perform SVD on A
        _, s, vt = np.linalg.svd(A)
 
        # The solution is the last row of V transposed (V^T), corresponding to the smallest singular valu
        #print("scaler_value: ",s[-1])
        #if(s[-1] < 5e-3):
        #    delete_idx.append(i)
        #    print("bad point")
        #    print("scaler_value: ",s[-1])
            #continue
        ret[i] = vt[3]
    #print("len before", ret.shape)

    #ret = np.delete(ret,delete_idx, axis=0)
    #print("len after", ret.shape)

    ret = ret / ret[:, 3, np.newaxis] 
    #print("delete index: ",delete_idx)
    #print(" ")
    # Return the 3D points in homogeneous coordinates
    return ret[:,:3],s


def rpy_to_quaternion(roll, pitch, yaw, degrees=False):

    r = R.from_euler('xyz', [roll, pitch, yaw], degrees=degrees)
    return r.as_quat() 

def intigrate(val1,val2,dt):
    return ((val2+val1)*dt) / 2
