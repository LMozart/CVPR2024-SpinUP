import torch
import cv2 as cv
import numpy as np
from m_utils import *
import torch.nn.functional as F


def erode_outer_contour(mask):
    dilation = cv.erode(mask, np.ones((3, 3)), iterations = 1)
    return dilation

def persp_contour_normal(idxp_contour, cnt_nml, f=50, H=512, W=512):
    uv = np.mgrid[:H, :W]
    uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
    uv = uv.reshape(2, -1).transpose(1, 0)                                             # [HW, 2]

    K = [[H / 36 * f,  0.,           H // 2.],
        [0.,           W / 36 * f,  W // 2.],
        [0.,           0.,           1.]]
    K = torch.from_numpy(np.array(K)).float()

    xyz = torch.cat((uv, torch.ones_like(uv[..., 0:1])), dim=-1).float()
    xyz = lift(xyz[..., 0, None], xyz[..., 1, None], xyz[..., 2, None], K[None, ...])[..., :3, 0]
    xyz = xyz.reshape(H, W, 3)

    xyz = F.normalize(xyz, p=2, dim=-1)
    v_xyz = xyz[idxp_contour]

    theta = torch.atan2(cnt_nml[idxp_contour[0], idxp_contour[1], 1], cnt_nml[idxp_contour[0], idxp_contour[1], 0])    
    phi   = torch.atan(-v_xyz[..., 2] / (v_xyz[..., 0] * torch.cos(theta) + v_xyz[..., 1] * torch.sin(theta)))
    phi[phi < 0] = np.pi + phi[phi < 0]

    cnt_x = torch.cos(theta) * torch.sin(phi)
    cnt_y = torch.sin(theta) * torch.sin(phi)
    cnt_z = torch.cos(phi)

    cnt_nmls = torch.stack((cnt_x, 
                            cnt_y, 
                            cnt_z), dim=-1)
    return cnt_nmls, v_xyz