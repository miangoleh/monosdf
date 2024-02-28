import numpy as np
import cv2
import torch
import os
from scipy.spatial.transform import Slerp
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import json
import trimesh
import glob
import PIL
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

import open3d as o3d

CAMERA_STR = ["camera1", "camera2", "camera3", "camera4", "camera5", "camera6", "camera7", "camera8"]

import sys
sys.path.append('../../../utils/')
from utils import loadYaml


# image [720, 1280]
# depth [720, 1280]
image_size = 384
trans_totensor = transforms.Compose([
    transforms.CenterCrop(2160),
    transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
])
depth_trans_totensor = transforms.Compose([
    transforms.CenterCrop(2160),
    transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
])


out_path_prefix = '../data/Studio/'
data_root = '/project/aksoy-lab/Mahdi/MultiViewFromMono/Mytestdata/'
scenes = ['frame_20000']
out_names = ['scan1']
metric_depth_path = '/project/aksoy-lab/Mahdi/MultiViewFromMono/Mytestdata/frame_20000/dense/stereo/depth_maps'

for scene, out_name in zip(scenes, out_names):
    out_path = os.path.join(out_path_prefix, out_name)
    os.makedirs(out_path, exist_ok=True)
    print(out_path)

    folders = ["image", "mask"]
    for folder in folders:
        out_folder = os.path.join(out_path, folder)
        os.makedirs(out_folder, exist_ok=True)

    # load color 
    color_path = os.path.join(data_root, scene, 'input')
    color_paths = sorted(glob.glob(os.path.join(color_path, '*.png')), 
        key=lambda x: int(os.path.basename(x)[:-4]))
    print(color_paths)

    # load mask 
    mask_path = os.path.join(data_root, scene, 'alpha')
    mask_paths = sorted(glob.glob(os.path.join(mask_path, '*.png')), 
        key=lambda x: int(os.path.basename(x)[:-4]))
    print(mask_paths)

    # load metric depth
    depth_paths = sorted(glob.glob(os.path.join(metric_depth_path, '*.npy')), 
        key=lambda x: int(os.path.basename(x)[:-4]))

    # load intrinsic
    camera_params = loadYaml(os.path.join(data_root,scene,"camera_parameters.yaml"))
    
    intrinsics = []
    for camera in CAMERA_STR:
        intrinsic_ = camera_params[camera]['intrinsics']
        intrinsics.append(np.array(intrinsic_))

    poses = []
    for camera in CAMERA_STR:
        extrinsics_ = camera_params[camera]['extrinsics']
        poses_ = np.linalg.inv(np.array(extrinsics_))
        poses.append(poses_)

    poses = np.array(poses) 

    # deal with invalid poses
    valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
    min_vertices = poses[:, :3, 3][valid_poses].min(axis=0)
    max_vertices = poses[:, :3, 3][valid_poses].max(axis=0)
 
    center = (min_vertices + max_vertices) / 2.
    scale = 2. / (np.max(max_vertices - min_vertices) + 0.1)
    print(center, scale)

    # we should normalized to unit cube
    scale_mat = np.eye(4).astype(np.float32)
    scale_mat[:3, 3] = -center
    scale_mat[:3 ] *= scale 
    scale_mat = np.linalg.inv(scale_mat)

    H, W = 2160, 4096
    new_intrinsics = []
    for camera_intrinsic in intrinsics:
        # center crop by 2160
        offset_x = (W - 2160) * 0.5
        offset_y = (H - 2160) * 0.5
        camera_intrinsic[0, 2] -= offset_x
        camera_intrinsic[1, 2] -= offset_y
        # resize
        resize_factor = 384 / 2160.
        camera_intrinsic[:2, :] *= resize_factor

        K_int = np.eye(4)
        K_int[:3, :3] = camera_intrinsic
        new_intrinsics.append(K_int)


    out_index = 0
    pcds = []
    cameras = {}

    for idx, (pose, K, image_path, mask_path) in enumerate(zip(poses, new_intrinsics, color_paths, mask_paths)):
        print(idx)

        target_image = os.path.join(out_path, "image/%06d.png"%(out_index))
        print(target_image)
        img = Image.open(image_path)
        img_tensor = trans_totensor(img)
        img_tensor.save(target_image)

        target_mask= os.path.join(out_path, "mask/%06d.png"%(out_index))
        # mask = (np.ones((image_size, image_size, 3)) * 255.).astype(np.uint8)
        mask = Image.open(mask_path)
        mask_tensor = trans_totensor(mask)
        mask_tensor.save(target_mask)

        np.save(os.path.join(out_path, "%06d_mask.npy"%(out_index)), np.array(mask_tensor))
        depth = np.load(depth_paths[idx])
        
        # scale the depth to match the world transformation
        depth = depth * scale 

        depth = Image.fromarray(depth)
        depth_tensor = trans_totensor(depth)

        np.save(os.path.join(out_path, "%06d_depthmetric.npy"%(out_index)), np.array(depth_tensor))

        # save pose
        camera_mat = K @ np.linalg.inv(pose)
        
        #cameras["scale_mat_%d"%(out_index)] = np.eye(4).astype(np.float32)
        cameras["scale_mat_%d"%(out_index)] = scale_mat
        cameras["world_mat_%d"%(out_index)] = camera_mat

        out_index += 1

        pose_norm = np.linalg.inv(pose) @ scale_mat
        pose_norm = np.linalg.inv(pose_norm)

        pcds.append(pose_norm[:3, 3])

    # save pcd
    pcds = np.stack(pcds)
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcds)
    o3d.io.write_point_cloud(os.path.join(out_path,f"camera_pose.ply"), pcd_o3d)

    np.savez(os.path.join(out_path, "cameras.npz"), **cameras)