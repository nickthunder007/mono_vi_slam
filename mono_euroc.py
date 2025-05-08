import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation as R
import os
from tqdm import tqdm
import torch
#import bundle_adjustment2 as ba
import time
from utils import utils
import open3d as o3d
from torchvision.transforms import Compose
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import time
import torch
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Load the DepthAnything model
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_vits14').to(device).eval()


transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

cam1_wrt_drone = np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                            [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                            [ -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                            [0.0, 0.0, 0.0, 1.0]])


rot_cam1_wrt_drone = cam1_wrt_drone[:3,:3]
trans_cam1_wrt_drone = np.expand_dims(cam1_wrt_drone[:3,3],axis=-1)

#rot_cam2_wrt_imu = cam2_wrt_world[:3,:3]
#trans_cam2_wrt_imu = np.expand_dims(cam2_wrt_world[:3,3],axis=-1)


def extract_features(image):
    """Detect and compute keypoints and descriptors using ORB."""
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(des1, des2):
    """Match features using Brute-Force Matcher with Hamming distance."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    #matches = [i for i in matches if i.distance < 20]
    return matches

def add_ones(pts):
    """Helper function to add a column of ones to a 2D array (homogeneous coordinates)."""
    return np.hstack([pts, np.ones((pts.shape[0], 1))])

def triangulate_points(pose1, pose2, pts1, pts2):
    # Initialize the result array to store the homogeneous coordinates of the 3D points
    ret = np.zeros((pts1.shape[0], 4))
    pose1 = np.vstack([pose1, [0, 0, 0, 1]])
    pose2 = np.vstack([pose2, [0, 0, 0, 1]])
 
    # Invert the camera poses to get the projection matrices
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
 
    # Loop through each pair of corresponding points
    for i, p in enumerate(zip(add_ones(pts1), add_ones(pts2))):
        # Initialize the matrix A to hold the linear equations
        #print(len(p[0]))
        A = np.zeros((4, 4))
 
        # Populate the matrix A with the equations derived from the projection matrices and the points
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
 
        # Perform SVD on A
        _, s, vt = np.linalg.svd(A)
        cond_number = s[0] / s[-1]
        #print("condition number: ", cond_number)
 
        # The solution is the last row of V transposed (V^T), corresponding to the smallest singular value
        ret[i] = vt[3]
    #print("before:\n",ret)
    ret = ret / ret[:, 3, np.newaxis] 
    # Return the 3D points in homogeneous coordinates
    return ret[:,:3]



# Load camera intrinsic parameters
K = np.array([[458.654, 0.00000, 367.215],  # Focal length (fx, fy) and principal point (cx, cy)``
              [0.00000, 457.296, 248.375],
              [0.00000, 0.00000, 1.00000]], dtype=float)

fx,fy,cx,cy = [458.654, 457.296, 367.215, 248.375]  #fu, fv, cu, cv

dir_path = "/home/nick/Downloads/V1_01_easy/mav0/"
img_path = dir_path + "cam0/data.csv"
img_path2 = dir_path + "cam1/data.csv"
imu_path = dir_path + "state_groundtruth_estimate0/data.csv"


df_img = pd.read_csv(img_path)
df_imu = pd.read_csv(imu_path)

comman_timestemp = set(df_img['#timestamp [ns]']).intersection(set(df_imu['#timestamp']))

df_img = df_img[df_img["#timestamp [ns]"].isin(comman_timestemp)]
df_imu = df_imu[df_imu["#timestamp"].isin(comman_timestemp)]


points_3D = np.zeros((1,3))
pose = np.zeros((1,3))
diff = 1
match_dist = 10
dense_3d_map = np.zeros((1,3))
dense_color_map = np.zeros((1,3))
frame = 1
done_imgs = 0
inference_wait = 7
current_inference = 0


#______________________________________LOOP__________________________________________________________________________
for j in range(120,len(df_imu)-5,5):#(df)-1
    print(j)
    i=j
    idx1 = i
    idx2 = i + diff

    R1 = R.from_quat([df_imu.iloc[idx1][' q_RS_w []'],
                      df_imu.iloc[idx1][' q_RS_x []'],
                      df_imu.iloc[idx1][' q_RS_y []'],
                      df_imu.iloc[idx1][' q_RS_z []']],scalar_first=True)
    rot_drone1_wrt_world = R1.as_matrix()

    trans_drone1_wrt_world = np.array([[df_imu.iloc[idx1][' p_RS_R_x [m]']],
                                        [df_imu.iloc[idx1][' p_RS_R_y [m]']],
                                        [df_imu.iloc[idx1][' p_RS_R_z [m]']]]).copy()

    R2 = R.from_quat([df_imu.iloc[idx2][' q_RS_w []'],
                      df_imu.iloc[idx2][' q_RS_x []'],
                      df_imu.iloc[idx2][' q_RS_y []'],
                      df_imu.iloc[idx2][' q_RS_z []']],scalar_first=True)
    rot_drone2_wrt_world = R2.as_matrix()


    trans_drone2_wrt_world = np.array([[df_imu.iloc[idx2][' p_RS_R_x [m]']],
                                        [df_imu.iloc[idx2][' p_RS_R_y [m]']],
                                        [df_imu.iloc[idx2][' p_RS_R_z [m]']]])
    
    rpy1 = R.from_matrix(rot_drone1_wrt_world)
    rpy2 = R.from_matrix(rot_drone2_wrt_world)
    


    trans_cam1_zero = np.zeros((3,1))
    trans_cam2_wrt_cam1 = rot_cam1_wrt_drone.T @ rot_drone1_wrt_world.T @ ((trans_drone2_wrt_world + rot_drone2_wrt_world @ trans_cam1_wrt_drone)
                                                                            - (trans_drone1_wrt_world + rot_drone1_wrt_world @ trans_cam1_wrt_drone))
    
    rot_cam1_zero = np.eye(3)
    rot_cam2_wrt_cam1 = rot_cam1_wrt_drone.T @ rot_drone1_wrt_world.T @ rot_drone2_wrt_world @ rot_cam1_wrt_drone

    rpy3 = R.from_matrix(rot_cam2_wrt_cam1)
    

    P1 = np.hstack((rot_cam1_zero, trans_cam1_zero))
    P2 = np.hstack((rot_cam2_wrt_cam1, trans_cam2_wrt_cam1))


    # Load images
    img1 = cv2.imread(img_path[:-4] + "/" + df_img.iloc[idx1]['filename'])
    img2 = cv2.imread(img_path[:-4] + "/" + df_img.iloc[idx2]['filename'])
  
    # Extract and match features
    kp1, des1 = extract_features(img1)
    kp2, des2 = extract_features(img2)
    matches = match_features(des1, des2)
    matches = [i for i in matches if i.distance < match_dist]

    """
    for m in matches:

        x,y = map(int, kp1[m.queryIdx].pt)
        img_draw1 = cv2.circle(img2, (x,y) , 5, (0, 255, 0), -1)

        #x,y = map(int, kp2[m.trainIdx].pt)
        #img_draw2 = cv2.circle(img2, (x,y) , 5, (0, 255, 0), -1)
    
    cv2.imwrite("cam1//orb_img" + str(i) + ".jpg", img_draw1)
    #cv2.imwrite("datasets//cam1/orb_img" + str(i+diff) + ".jpg", img_draw2)
    """
    if(len(matches) < 1):
        continue

    # Convert matched keypoints to numpy arrays
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])  # Shape (N, 3)
    pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])

    K_inv = np.linalg.inv(K)
    # Multiply by inverse intrinsic matrix to get normalized coordinates
    pts1_norm = (K_inv @ pts1_h.T).T  # Shape: (N, 3)
    pts2_norm = (K_inv @ pts2_h.T).T


    x = np.round(pts1[:, 0]).astype(int)
    y = np.round(pts1[:, 1]).astype(int)

    points_wrt_cam1 = triangulate_points(P1, P2, pts1_norm, pts2_norm)
    feature_color = img1[y, x, :]  # shape: (N, 3)


    
    points_wrt_world = rot_drone1_wrt_world @ rot_cam1_wrt_drone @ points_wrt_cam1.T + (trans_drone1_wrt_world + rot_drone1_wrt_world @ trans_cam1_wrt_drone)
    points = points_wrt_world.T

    #rot_world_wrt_cam1 = (rot_cam1_wrt_drone.T @ rot_drone1_wrt_world.T)
    #rot_world_wrt_cam1_ = R.from_matrix(rot_world_wrt_cam1)
    #rodrig_world_wrt_cam1 = rot_world_wrt_cam1_.as_rotvec()
 
    #trans_cam1_wrt_world = -rot_world_wrt_cam1 @ (trans_drone1_wrt_world +
    #                                              rot_drone1_wrt_world @ trans_cam1_wrt_drone)
    start_time = time.time()

    img_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) / 255.0
    img_transformed = transform({'image': img_rgb})['image']
    img_tensor = torch.from_numpy(img_transformed).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = depth_anything(img_tensor)
    
    depth = F.interpolate(depth[None], (img1.shape[0], img1.shape[1]), mode='bilinear', align_corners=False)[0, 0]
    
    #depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.cpu().numpy().astype(np.float16)
    depth = np.clip(depth, 1, 15, out=None)

    depth = 1/depth

    depth = np.clip(depth, 0, 0.13, out=None)
    
    px_list = []
    for h in matches:
        
        x,y = kp1[h.queryIdx].pt
        px_list.append([int(x),int(y)])


    px_list = np.array(px_list)
    x_coord = px_list[:,1]
    y_coord = px_list[:,0]


    depth_vals = depth[x_coord, y_coord]
    trig_vals = points_wrt_cam1[:,2]
    scale = trig_vals/depth_vals


    dense_color = img1.reshape(-1,3)

    dense_depth = depth * scale.mean()
    dense_3d_cloud_ = utils.depth_to_3d(dense_depth,fx,fy,cx,cy)

    dense_3d_cloud = dense_3d_cloud_[dense_3d_cloud_[:,2] < dense_3d_cloud_[:,2].max()]
    dense_color = dense_color[dense_3d_cloud_[:,2] < dense_3d_cloud_[:,2].max()]

    

    
    #print("depth_vals: ",depth_vals, depth_vals.shape)
    #print("trig vals: ", trig_vals, trig_vals.shape)
    #print("scale: ", scale)
    #print("mean scale: ", scale.mean(), "std: ",scale.std())
    current_inference += 1
    print("current_inference: ", current_inference)

    if(scale.std() > 5.0 or current_inference < inference_wait):
        print("frame: ", j)
        continue

    for m in matches:

        x,y = map(int, kp1[m.queryIdx].pt)
        img_draw1 = cv2.circle(img2, (x,y) , 5, (0, 255, 0), -1)

        #x,y = map(int, kp2[m.trainIdx].pt)
        #img_draw2 = cv2.circle(img2, (x,y) , 5, (0, 255, 0), -1)
    
    cv2.imwrite("cam1//orb_img" + str(i) + ".jpg", img_draw1)
    current_inference = 0
    print("mean scale: ", scale.mean(), "std: ",scale.std())
    print("")
    done_imgs += 1


    dense_3d_cloud = dense_3d_cloud[::50,:]
    dense_color = dense_color[::50,:]/255.0

    dense_3d_cloud_wrt_w = rot_drone1_wrt_world @ rot_cam1_wrt_drone @ dense_3d_cloud.T + (trans_drone1_wrt_world + rot_drone1_wrt_world @ trans_cam1_wrt_drone)
    dense_3d_cloud_wrt_w = dense_3d_cloud_wrt_w.T

    print("dense 2d clowd wrt w", dense_3d_cloud_wrt_w.shape)
    print("dense colors: ", dense_color.shape)

    dense_3d_cloud_wrt_w_ = dense_3d_cloud_wrt_w.copy()
    dense_3d_cloud_wrt_w = dense_3d_cloud_wrt_w[dense_3d_cloud_wrt_w[:,2] > 0]
    dense_color = dense_color[dense_3d_cloud_wrt_w_[:,2] > 0]

    
    dense_3d_map = np.vstack([dense_3d_map, dense_3d_cloud_wrt_w])
    dense_color_map = np.vstack([dense_color_map, dense_color])
   
    
    print("time_taken: ",time.time() - start_time)

    
    points_3D = np.vstack([points_3D,points.copy()])
    pose = np.vstack([pose,trans_drone1_wrt_world.T])
    frame += 1

print("dense_3d_map: ",dense_3d_map.shape)
print("Recovered 3D points:\n", points_3D.shape)
print("pose_shape: ",pose.shape)
print("done_imgs: ",done_imgs)


#pcd1 = o3d.geometry.PointCloud()
#pcd1.points = o3d.utility.Vector3dVector(points_wrt_cam1)
#pcd1.colors = o3d.utility.Vector3dVector(feature_color/255.0)
#o3d.visualization.draw_geometries([pcd1])
#o3d.io.write_point_cloud("/home/nick/Downloads/thesis_images/code_imgs/mono_eurec_triang_1.ply", pcd1)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(dense_3d_map)
pcd.colors = o3d.utility.Vector3dVector(dense_color_map)
o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud("/home/nick/Downloads/thesis_images/code_imgs/mono_eurec_room6.ply", pcd)  # or .pcd, .xyz

exit()

# Sample small point cloud (N, 3)
points = dense_3d_map

# Sample RGB colors (N, 3), normalized to [0, 1]
colors = dense_color_map

# Create 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot point cloud
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=20)

# Add grid, axis labels, and title
ax.grid(True)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('Colored Point Cloud')

plt.tight_layout()
plt.show()