import cv2 as cv
import numpy as np
import torch


def compute_contour_normal(_mask):
    blur = cv.GaussianBlur(_mask, (11, 11), 0)
    n_x = -cv.Sobel(blur, cv.CV_32F, 1, 0, ksize=11, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    n_y = -cv.Sobel(blur, cv.CV_32F, 0, 1, ksize=11, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)

    n = np.sqrt(n_x**2 + n_y**2) + 1e-5
    contour_normal = np.zeros((_mask.shape[0], _mask.shape[1], 3), np.float32)
    contour_normal[:, :, 0] = n_x / n
    contour_normal[:, :, 1] = -n_y / n
    contour_normal = torch.tensor(contour_normal, dtype=torch.float32)
    # res_nml[..., 1:] = -res_nml[..., 1:]
    
    return contour_normal

def get_contour_idx(mask):
    mask_x1, mask_x2, mask_y1, mask_y2 = mask.copy(), mask.copy(), mask.copy(), mask.copy()
    mask_x1[:-1, :] = mask[1:, :]
    mask_x2[1:, :]  = mask[:-1, :]
    mask_y1[:, :-1] = mask[:, 1:]
    mask_y2[:, 1:]  = mask[:, :-1]
    mask_1 = mask_x1 * mask_x2 * mask_y1 * mask_y2
    idxp_contour = np.where((mask_1 < 0.5) & (mask > 0.5))
    return idxp_contour