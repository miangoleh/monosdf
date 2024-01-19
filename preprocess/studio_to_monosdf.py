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

CAMERA_STR = ["camera1", "camera2", "camera3", "camera4", "camera5", "camera6", "camera7", "camera8"]

import sys
sys.path.append('../../../utils/')
from utils import loadYaml


# image [720, 1280]
# depth [720, 1280]
image_size = 384
trans_totensor = transforms.Compose([
    transforms.CenterCrop(720),
    transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
])
depth_trans_totensor = transforms.Compose([
    transforms.CenterCrop(720),
    transforms.Resize(image_size, interpolation=PIL.Image.NEAREST),
])


out_path_prefix = '../data/Studio/'
data_root = '/project/aksoy-lab/Mahdi/MultiViewFromMono/Mytestdata/'
scenes = ['frame_20000']
out_names = ['scan1']

for scene, out_name in zip(scenes, out_names):
    out_path = os.path.join(out_path_prefix, out_name)
    os.makedirs(out_path, exist_ok=True)
    print(out_path)

    folders = ["image", "mask", "depth"]
    for folder in folders:
        out_folder = os.path.join(out_path, folder)
        os.makedirs(out_folder, exist_ok=True)

    # load color 
    color_path = os.path.join(data_root, scene, 'input')
    color_paths = sorted(glob.glob(os.path.join(color_path, '*.png')), 
        key=lambda x: int(os.path.basename(x)[:-4]))
    print(color_paths)

    # load intrinsic
    camera_params = loadYaml(os.path.join(data_root,scene,"camera_parameters.yaml"))
    
    import ipdb; ipdb.set_trace()
    intrinsics = []
    for camera in CAMERA_STR:
        intrinsic_ = camera_params[camera]['intrinsics']
        intrinsics.append(np.array(intrinsic_))

    extrinsics = []
    poses = []
    for camera in CAMERA_STR:
        extrinsics_ = camera_params['extrinsics']
        extrinsics.append(np.array(extrinsics_))
        poses_ =  np.linalg.inv(np.array(extrinsics_))
        poses.append(poses_)

    poses = np.array(poses)

    # deal with invalid poses
    valid_poses = np.isfinite(poses).all(axis=2).all(axis=1)
    min_vertices = poses[:, :3, 3][valid_poses].min(axis=0)
    max_vertices = poses[:, :3, 3][valid_poses].max(axis=0)
 
    center = (min_vertices + max_vertices) / 2.
    scale = 2. / (np.max(max_vertices - min_vertices) + 3.)
    print(center, scale)

    # we should normalized to unit cube
    scale_mat = np.eye(4).astype(np.float32)
    scale_mat[:3, 3] = -center
    scale_mat[:3 ] *= scale 
    scale_mat = np.linalg.inv(scale_mat)

    H, W = 2160, 4096
    new_intrinsics = []
    for camera_intrinsic in intrinsics:
        print(camera_intrinsic)
        # center crop by 2160
        offset_x = (W - 2160) * 0.5
        offset_y = (H - 2160) * 0.5
        camera_intrinsic[0, 2] -= offset_x
        camera_intrinsic[1, 2] -= offset_y
        # resize
        resize_factor = 384 / 2160.
        camera_intrinsic[:2, :] *= resize_factor

        K = np.eye(4)
        K[:3, :3] = camera_intrinsic
        new_intrinsics.append(K)


    out_index = 0
    pcds = []
    cameras = {}

    for idx, (pose, extrinsic, K, image_path) in enumerate(zip(poses, extrinsics, new_intrinsics, color_paths)):
        print(idx)

        target_image = os.path.join(out_path, "image/%06d.png"%(out_index))
        print(target_image)
        img = Image.open(image_path)
        img_tensor = trans_totensor(img)
        img_tensor.save(target_image)

        mask = (np.ones((image_size, image_size, 3)) * 255.).astype(np.uint8)

        target_image = os.path.join(out_path, "mask/%03d.png"%(out_index))
        cv2.imwrite(target_image, mask)
        
        # save pose
        pcds.append(pose[:3, 3])
        pose = K @ extrinsics
        
        #cameras["scale_mat_%d"%(out_index)] = np.eye(4).astype(np.float32)
        cameras["scale_mat_%d"%(out_index)] = scale_mat
        cameras["world_mat_%d"%(out_index)] = pose

        out_index += 1

    np.savez(os.path.join(out_path, "cameras.npz"), **cameras)