import cv2 as cv
import os
import numpy as np
import torch
import scipy.io as scio
from os.path import join as pjoin
import glob
import scipy.io as sio
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def parse_txt(filename):
    out_list = []
    with open(filename) as f:
        lines = [line.rstrip() for line in f]
        for x in lines:
            lxyz = np.array([float(v) for v in x.strip().split()], dtype=np.float32)
            out_list.append(lxyz)
    out_arr = np.stack(out_list, axis=0).astype(np.float32)
    return out_arr


def read_mat_file(filename):
    """
    :return: Normal_ground truth in shape: (height, width, 3)
    """
    mat = sio.loadmat(filename)
    gt_n = mat['Normal_gt']
    return gt_n.astype(np.float32)


def normal_to_mask(normal, thres=0.9):
    mask = (np.square(normal).sum(2) > thres).astype(np.float32)
    return mask

def load_nml_map(path, rescale=None):
    """Parse normal map, convert the value range from [0, 1] to [-1, 1].
    """
    nml = cv.imread(pjoin(path,"nml.exr"), -1)
    nml = cv.cvtColor(nml, cv.COLOR_BGR2RGB)
    if rescale:
        nml = cv.resize(nml, rescale, interpolation = cv.INTER_NEAREST)
    nml = 2 * nml - 1
    return nml

def load_natural(path, lgt_path, cfg=None, test=False, scale=1):
    images = []
    h = 512 // scale
    for img_file in sorted(glob.glob(pjoin(path,"obj","*.png"))):
        img = cv.imread(img_file)
        img = cv.cvtColor(img[..., :3], cv.COLOR_BGR2RGB)
        images.append((img / 255.))
    images = np.stack(images, axis=0)

    lgt_images = []
    for img_file in sorted(glob.glob(pjoin(path,"env","*.png"))):
        img = cv.imread(img_file, -1)
        img = cv.cvtColor(img[..., :3], cv.COLOR_BGR2RGB)
        img = cv.resize(img, (h * 2, h), interpolation = cv.INTER_NEAREST)
        lgt_images.append(img / 255)
    lgt_images = np.stack(lgt_images, axis=0)

    normal = load_nml_map(pjoin(path, "nml"))
    gt_normal = normal
    norm = np.sqrt(gt_normal * gt_normal).sum(2, keepdims=True)
    gt_normal = gt_normal / (norm + 1e-8)
    mask = normal_to_mask(normal)

    # Load Light SG
    lgtSG  = np.load(pjoin(path, lgt_path))
    gt_rot = np.linspace(0, (images.shape[0]-1)/images.shape[0] * 2 * np.pi, images.shape[0])

    # Load Calibrate Parameters
    cam_dict = np.load(pjoin(path, "camera_params.npy"), allow_pickle=True).item()
    cam_dict["K"][0, 0] = cam_dict["K"][0, 0] / scale
    cam_dict["K"][1, 1] = cam_dict["K"][1, 1] / scale
    cam_dict["K"][0, 2] = cam_dict["K"][0, 2] / scale
    cam_dict["K"][1, 2] = cam_dict["K"][1, 2] / scale
    cam_dict["fs"] = 36

    out_dict = {'images': images, 'mask': mask, 'gt_normal': gt_normal, 'lgt': lgt_images, "lgtSG": lgtSG, "cam_params": cam_dict, "gt_rot": gt_rot}
    return out_dict


def load_real(path, lgt_path, cfg=None, test=False, scale=1):
    images = []
    mask      = cv.imread(pjoin(path,"mask.png"))

    for img_file in sorted(glob.glob(pjoin(path, "obj", "*.png"))):
        img = cv.imread(img_file)
        img = cv.cvtColor(img[..., :3], cv.COLOR_BGR2RGB)
        img = cv.resize(img, (540, 540), interpolation = cv.INTER_NEAREST)
        # Bugging remove
        img = cv.copyMakeBorder(img, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=(0, 0, 0))
        images.append((img / 255.) ** 2.2)
    images = np.stack(images, axis=0)

    lgt_images = np.ones((images.shape[0], 256, 512, 3))

    gt_normal = np.zeros_like(img)
    
    if len(mask.shape) == 3:
        mask = mask[..., 0]
    mask      = cv.resize(mask, (540, 540), interpolation = cv.INTER_NEAREST)
    mask      = cv.copyMakeBorder(mask, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=(0, 0, 0)) / 255.
    mask[mask > 0.5] = 1.
    mask = mask.astype(np.float32)
    assert mask.shape[0] == images.shape[1]
    assert mask.shape[1] == images.shape[2]

    # Load Light SG
    if not test:
        lgtSG = np.load(pjoin(path, lgt_path))
    else:
        lgtSG = np.random.rand(64, 7)

    # Load Calibrate Parameters
    cam_dict = np.load(pjoin(path, "camera_params.npy"), allow_pickle=True).item()
    cam_dict["K"][0, 0] = cam_dict["K"][0, 0] / scale
    cam_dict["K"][1, 1] = cam_dict["K"][1, 1] / scale
    cam_dict["K"][0, 2] = cam_dict["K"][0, 2] / scale
    cam_dict["K"][1, 2] = cam_dict["K"][1, 2] / scale
    if "fs" not in cam_dict.keys():
        cam_dict["fs"] = 14.9
    if images.shape[1] != 540:
        cam_dict["K"][0, 0] = cam_dict["K"][0, 0] / 512 * images.shape[1]
        cam_dict["K"][1, 1] = cam_dict["K"][1, 1] / 512 * images.shape[1]
        cam_dict["K"][0, 2] = images.shape[1] / 2
        cam_dict["K"][1, 2] = images.shape[1] / 2
    
    gt_rot = np.linspace(0, (images.shape[0]-1)/images.shape[0] * 2 * np.pi, images.shape[0])
    cam_dict["R"] = gt_rot

    out_dict = {'images': images, 'mask': mask, 'gt_normal': gt_normal, 'lgt': lgt_images, "lgtSG": lgtSG, "cam_params": cam_dict, "gt_rot": gt_rot}
    return out_dict


def load_unitsphere():
    mask_files = os.path.join("./data/DiLiGenT/pmsData/ballPNG", "mask.png")
    mask = cv.imread(mask_files, 0).astype(np.float32) / 255.
    gt_normal_files = os.path.join("./data/DiLiGenT/pmsData/ballPNG", "Normal_gt.mat")
    gt_normal = read_mat_file(gt_normal_files)

    return {'mask': mask, 'gt_normal': gt_normal}
