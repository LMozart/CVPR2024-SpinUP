import torch
import torch.nn.functional as F
import torchvision
import os
import cv2 as cv
import math
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def cal_dirs_acc(gt_l, pred_l):
    dot_product = (gt_l * pred_l).sum(-1).clamp(-1, 1)
    angular_err = torch.acos(dot_product) * 180.0 / math.pi
    l_err_mean = angular_err.mean()
    return l_err_mean.item(), angular_err


def cal_ints_acc(gt_i, pred_i):
    # Red channel:
    gt_i_c = gt_i[:, :1]
    pred_i_c = pred_i[:, :1]
    scale = torch.linalg.lstsq(pred_i_c, gt_i_c).solution
    ints_ratio1 = (gt_i_c - scale * pred_i_c).abs() / (gt_i_c + 1e-8)
    # Green channel:
    gt_i_c = gt_i[:, 1:2]
    pred_i_c = pred_i[:, 1:2]
    scale = torch.linalg.lstsq(pred_i_c, gt_i_c).solution
    ints_ratio2 = (gt_i_c - scale * pred_i_c).abs() / (gt_i_c + 1e-8)
    # Blue channel:
    gt_i_c = gt_i[:, 2:3]
    pred_i_c = pred_i[:, 2:3]
    scale = torch.linalg.lstsq(pred_i_c, gt_i_c).solution
    ints_ratio3 = (gt_i_c - scale * pred_i_c).abs() / (gt_i_c + 1e-8)

    ints_ratio = (ints_ratio1 + ints_ratio2 + ints_ratio3) / 3
    return ints_ratio.mean().item(), ints_ratio.mean(dim=-1)


def plot_images(filename, pixels, h, w, channel, idxp, idxp_invalid, bounding_box_int):
    img = torch.ones((h, w, channel), device=pixels.device)
    img[idxp] = pixels
    img = img.cpu().numpy()
    img = np.clip(img * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
    img[idxp_invalid] = 255
    img = img[bounding_box_int[0]:bounding_box_int[1], bounding_box_int[2]:bounding_box_int[3]]
    cv.imwrite(filename, img)

def process_images(pixels, h, w, channel, idxp, idxp_invalid, bounding_box_int):
    img = torch.ones((h, w, channel), device=pixels.device)
    img[idxp] = pixels
    img = img.cpu().numpy()
    img = np.clip(img * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
    img[idxp_invalid] = 255
    img = img[bounding_box_int[0]:bounding_box_int[1], bounding_box_int[2]:bounding_box_int[3]]
    return img


def writer_add_image(file_name, epoch, writer):
    if writer is None:
        return
    img = cv.imread(file_name)
    img = torch.tensor(img[:, :, ::-1] / 255.)
    img_grid = torchvision.utils.make_grid(img.permute(2, 0, 1)[None, ...])
    basename = os.path.basename(file_name)[:-4]
    writer.add_image(basename, img_grid, epoch)
    return

def dot(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum(x * y, axis=-1, keepdims=True)


def magnitude(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), 1e-12))


def normalize(x: np.ndarray) -> np.ndarray:
    return x / magnitude(x)


def axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])