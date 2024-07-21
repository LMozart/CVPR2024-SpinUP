import cv2 as cv
import os
import numpy as np
import torch
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

def erode_outer_contour(mask):
    dilation = cv.erode(mask, np.ones((3, 3)), iterations = 1)
    return dilation

def load_natural(path, cfg=None, erode_mask=False):
    images = []
    for img_file in sorted(glob.glob(pjoin(path,"obj","*.png"))):
        img = cv.imread(img_file)
        img = cv.cvtColor(img[..., :3], cv.COLOR_BGR2RGB)
        img = cv.resize(img, (512, 512), interpolation = cv.INTER_NEAREST)
        images.append(img / 255.)
    images = np.stack(images, axis=0)
    
    lgt_images = []
    for img_file in sorted(glob.glob(pjoin(path,"env","*.png"))):
        img = cv.imread(img_file)
        img = cv.cvtColor(img[..., :3], cv.COLOR_BGR2RGB)
        img = cv.resize(img, (512, 256), interpolation = cv.INTER_NEAREST)
        lgt_images.append(img)
    lgt_images = np.stack(lgt_images, axis=0)

    normal = load_nml_map(pjoin(path, "nml"), (512, 512))
    gt_normal = normal
    gt_normal[..., 0] = -gt_normal[..., 0]
    gt_normal[..., 2] = -gt_normal[..., 2]
    norm = np.sqrt(gt_normal * gt_normal).sum(2, keepdims=True)
    gt_normal = gt_normal / (norm + 1e-8)
    mask = normal_to_mask(normal)
    if erode_mask:
        mask = erode_outer_contour(mask)

    out_dict = {'images': images, 'mask': mask, 'gt_normal': gt_normal, 'lgt': lgt_images}
    return out_dict

def load_real(path, cfg=None, test=False, scale=1, erode_mask=True):
    images = []
    for img_file in sorted(glob.glob(pjoin(path, "obj", "*.png"))):
        img = cv.imread(img_file)
        img = cv.cvtColor(img[..., :3], cv.COLOR_BGR2RGB)
        # print(img.shape)
        img = cv.resize(img, (540, 540), interpolation = cv.INTER_NEAREST)
        # Bugging remove
        # img = cv.copyMakeBorder(img, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=(0, 0, 0))
        images.append((img / 255.) ** 2.2)
        # images.append((img / 255.))
    images = np.stack(images, axis=0)

    lgt_images = np.ones((images.shape[0], 256, 512, 3))
    mean_imgs = images.mean(0)

    gt_normal = np.zeros_like(img)
    mask      = cv.imread(pjoin(path,"mask.png"))[..., 0]
    mask = cv.resize(mask, (540, 540), interpolation = cv.INTER_NEAREST)
    # mask      = cv.copyMakeBorder(mask, 1, 1, 1, 1, cv.BORDER_CONSTANT, value=(0, 0, 0))
    mask = mask.astype(np.float32)
    if erode_mask:
        mask = erode_outer_contour(mask)

    # Load Calibrate Parameters
    cam_dict = np.load(pjoin(path, "camera_params.npy"), allow_pickle=True).item()
    cam_dict["K"][0, 0] = cam_dict["K"][0, 0] / scale
    cam_dict["K"][1, 1] = cam_dict["K"][1, 1] / scale
    cam_dict["K"][0, 2] = cam_dict["K"][0, 2] / scale
    cam_dict["K"][1, 2] = cam_dict["K"][1, 2] / scale
    if "fs" not in cam_dict.keys():
        cam_dict["fs"] = 14.9
    if images.shape[1] == 540:
        cam_dict["K"][0, 0] = cam_dict["K"][0, 0] / 512 * 540
        cam_dict["K"][1, 1] = cam_dict["K"][1, 1] / 512 * 540
        cam_dict["K"][0, 2] = 270
        cam_dict["K"][1, 2] = 270
    cam_dict["R"] = np.linspace(0, (images.shape[0]-1)/images.shape[0] * 2 * np.pi, images.shape[0])

    out_dict = {'images': images, 'mask': mask, 'gt_normal': gt_normal, 'lgt': lgt_images, "cam_params": cam_dict}
    return out_dict