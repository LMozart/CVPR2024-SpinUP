import argparse
import torch
from scipy.interpolate import griddata
from load_natural import load_natural, load_real
from m_utils      import *
from pre_utils    import *
from cnt_utils    import *
from sphericalHarmonics import *
import random

import torch.nn as nn
from   tqdm     import trange
import imageio
import os
import cv2 as cv

PANO_W = 256
PANO_H = 128
L_MAX  = 3
erode_mask = True
min_thresh = 0
max_thresh = 80


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def _normalize(x: np.ndarray) -> np.ndarray:
    return x / _magnitude(x)

def _dot(x, y):
    return np.sum(x * y, axis=-1, keepdims=True)

def _magnitude(x: torch.Tensor) -> torch.Tensor:
    return _safe_sqrt(_dot(x, x))

def _safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    sqrt_in = np.maximum(x, np.ones_like(x) * 0.00001)
    return np.sqrt(sqrt_in)

def safe_init(num_sgs, mode="uniform"):
    if mode=="uniform":
        axis = []
        inc = np.pi * (3.0 - np.sqrt(5.0))
        off = 2.0 / num_sgs
        for k in range(num_sgs):
            y = k * off - 1.0 + (off / 2.0)
            r = np.sqrt(1.0 - y * y)
            phi = k * inc
            axis.append(_normalize(np.array([np.sin(phi) * r, y, -np.cos(phi) * r])))

        minDp = 1.0
        for a in axis:
            h = _normalize(a + axis[0])
            minDp = min(minDp, _dot(h, axis[0]))

        sharpness = (np.log(0.65) * num_sgs) / (minDp - 1.0)
        print(num_sgs / (minDp - 1.0))

        lobe = np.stack(axis, 0)  # Shape: num_sgs, 3
        mu   = np.ones_like(axis)
        lbd  = np.ones((num_sgs, 1)) * sharpness
    elif mode=="random":
        lobe = np.random.uniform(-1, 1, (num_sgs, 3))      # [N_l, 3]
        mu   = np.random.uniform(0, 1, (num_sgs, 3))       # [N_l, 3]
        lbd  = np.random.uniform(1, 2, (num_sgs, 1)) # [N_l, 3]
        lobe = _normalize(lobe)                             # [N_l, 3]
    
    res_dict = {"lobe": lobe,
                "mu": mu,
                "lbd": lbd}
    return res_dict

def fill_hole_with_nearest(vecs, std_vecs, std_rgb):
    rgb = torch.zeros_like(vecs)
    for i in range(vecs.shape[0]):
        cos_val = (vecs[i, None, :] * std_vecs).sum(-1)
        idx = torch.argmax(cos_val, dim=0)
        rgb[i, :] = std_rgb[idx, :]
    return rgb

def SG2Envmap(lgtSGs, H, W, upper_hemi=False):
    # exactly same convetion as Mitsuba, check envmap_convention.png
    if upper_hemi:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(-np.pi, np.pi, W)])
    else:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(-np.pi, np.pi, W)])

    viewdirs = torch.stack([torch.sin(theta) * torch.sin(phi),
                             torch.cos(phi),
                             -torch.cos(theta) * torch.sin(phi)], dim=-1)

    # lgtSGs = lgtSGs.clone().detach()
    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]
    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    lgtSGs = lgtSGs.view([1,]*len(dots_sh)+[M, 5]).expand(dots_sh+[M, 5])
    # sanity
    # [..., M, 3]
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + 1e-5)
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., 4:])  # positive values
    # [..., M, 3]
    rgb = lgtSGMus * torch.exp(lgtSGLambdas * (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]
    envmap = rgb.reshape((H, W))
    return envmap

def blurIBL(ibl, amount=5):
	x = ibl.copy()
	x[:,:,0] = ndimage.gaussian_filter(ibl[:,:,0], sigma=amount)
	x[:,:,1] = ndimage.gaussian_filter(ibl[:,:,1], sigma=amount)
	x[:,:,2] = ndimage.gaussian_filter(ibl[:,:,2], sigma=amount)
	return x

def fit_env_map(cfg):
    device = torch.device(f"cuda:{cfg.gpu}")
    if "Real" not in cfg.data_path:
        data_dict = load_natural(cfg.data_path, erode_mask=erode_mask)
    else:
        data_dict = load_real(cfg.data_path, erode_mask=erode_mask)
    cam_params= np.load(os.path.join(cfg.data_path, "camera_params.npy"), allow_pickle=True).item()
    euler     = cam_params["R"].reshape(-1)
    
    mask = data_dict["mask"]
    imgs = torch.from_numpy(data_dict["images"])
    
    # Orthogonal Contour Map
    cnt_nml_map = compute_contour_normal(mask)
    # cnt_nml_map = normal
    cnt_nml_map = cnt_nml_map / (magnitude(cnt_nml_map) + 1e-5)
    pre_cnt_idx = get_contour_idx(mask)

    # Perspective Contour Map
    print(imgs.shape)
    persp_cnt_nml, v_xyz = persp_contour_normal(cnt_nml      = cnt_nml_map,
                                                idxp_contour = pre_cnt_idx,
                                                H = imgs.shape[1],
                                                W = imgs.shape[2])
    v_xyz = v_xyz / (magnitude(v_xyz) + 1e-5)
    persp_cnt_nml = persp_cnt_nml / (magnitude(persp_cnt_nml) + 1e-5)
    
    # Update Contour Normal Map with the Perspective Contour Normal
    cnt_nml_map[pre_cnt_idx[0], pre_cnt_idx[1]] = persp_cnt_nml
    cnt_nml_map = cnt_nml_map / (magnitude(cnt_nml_map) + 1e-5)
    
    # Get Half Normals 
    right_cnt_nml_map = cnt_nml_map.clone()
    left_cnt_nml_map  = cnt_nml_map.clone()
    right_cnt_nml_map[right_cnt_nml_map[..., 0] > 0., :] = 0.
    left_cnt_nml_map[left_cnt_nml_map[..., 0] < 0., :]   = 0.

    mask_half = np.zeros_like(cnt_nml_map)
    mask_half[pre_cnt_idx] = 1.

    right_cnt_nml_map = right_cnt_nml_map * mask_half
    left_cnt_nml_map  = left_cnt_nml_map * mask_half
    right_cnt_idx     = np.where(magnitude(right_cnt_nml_map)[..., 0] > 0.1)
    left_cnt_idx      = np.where(magnitude(left_cnt_nml_map)[..., 0] > 0.1)

    # Get Pretrained Data.
    right_valid_nml = cnt_nml_map[right_cnt_idx]
    left_valid_nml  = cnt_nml_map[left_cnt_idx]
    right_valid_rgb = imgs[:, right_cnt_idx[0], right_cnt_idx[1]]
    left_valid_rgb  = imgs[:, left_cnt_idx[0], left_cnt_idx[1]]

    # gt
    if os.path.exists(os.path.join(cfg.data_path, "env/000.png")):
        gt_img = cv2.imread(os.path.join(cfg.data_path, "env/000.png"))
    else:
        gt_img = np.zeros((PANO_W, PANO_H, 3))
    gt_img = cv.resize(gt_img, (PANO_W, PANO_H), interpolation = cv.INTER_NEAREST)
    gt_img = torch.from_numpy(gt_img) / 255.
    
    # Euler for rotation.
    rot_mats  = axis_angle_rotation(torch.from_numpy(euler)).float()
    p_right_valid_nml = (rot_mats[:, None] @ right_valid_nml[None, ..., None])[..., 0]
    rot_mats  = axis_angle_rotation(torch.from_numpy(euler)).float()
    p_left_valid_nml = (rot_mats[:, None] @ left_valid_nml[None, ..., None])[..., 0]

    r_idx = np.unravel_index(right_valid_rgb.mean(-1).argmax(), right_valid_rgb.mean(-1).shape)
    l_idx = np.unravel_index(left_valid_rgb.mean(-1).argmax(), left_valid_rgb.mean(-1).shape)
    r_ax = torch.nn.functional.normalize(p_right_valid_nml[r_idx[0], r_idx[1], 0:2], p=2, dim=-1)
    l_ax = torch.nn.functional.normalize(p_left_valid_nml[l_idx[0], l_idx[1], 0:2], p=2, dim=-1)
    # fix_euler = 0
    fix_euler = torch.acos((r_ax * l_ax).sum())
    print(fix_euler / np.pi * 180)
    
    rot_mats  = axis_angle_rotation(torch.from_numpy(euler) + fix_euler / 2).float()
    right_valid_nml = (rot_mats[:, None] @ right_valid_nml[None, ..., None])[..., 0]
    rot_mats  = axis_angle_rotation(torch.from_numpy(euler) - fix_euler / 2).float()
    left_valid_nml = (rot_mats[:, None] @ left_valid_nml[None, ..., None])[..., 0]


    # Intensity Profile
    right_valid_int = right_valid_rgb.mean(-1)
    right_T_pmax = np.percentile(right_valid_int, max_thresh, axis=-2)
    right_T_pmin = np.percentile(right_valid_int, min_thresh, axis=-2)
    left_valid_int = left_valid_rgb.mean(-1)
    left_T_pmax = np.percentile(left_valid_int, max_thresh, axis=-2)
    left_T_pmin = np.percentile(left_valid_int, min_thresh, axis=-2)
    diff_rgb_list = []
    diff_nml_list = []
    diff_inv_rgb_list = []
    diff_inv_nml_list = []

    for i in range(right_T_pmax.shape[0]):
        diff_idx = np.where(np.logical_and(right_valid_int[:, i]>=right_T_pmin[i], right_valid_int[:, i]<=right_T_pmax[i]))
        diff_rgb_list.append(right_valid_rgb[diff_idx[0], i, :])
        diff_nml_list.append(right_valid_nml[diff_idx[0], i, :])
        inv_diff_idx = np.where(np.logical_or(right_valid_int[:, i]<right_T_pmin[i], right_valid_int[:, i]>right_T_pmax[i]))
        test_inv_diff_idx = inv_diff_idx[0].copy()
        test_inv_diff_idx = test_inv_diff_idx.reshape(1, -1).repeat(len(diff_idx[0]), 0)
        test_inv_diff_idx = np.abs(diff_idx[0].reshape(-1, 1) - test_inv_diff_idx)
        idx_of_diff_idx = np.argmin(test_inv_diff_idx, axis=0)
        diff_inv_idx = []
        for j in range(len(idx_of_diff_idx)):
            diff_inv_idx.append(diff_idx[0][idx_of_diff_idx[j]])
        diff_inv_rgb_list.append(right_valid_rgb[diff_inv_idx, i, :])
        diff_inv_nml_list.append(right_valid_nml[inv_diff_idx[0], i, :])
    for i in range(left_T_pmax.shape[0]):
        diff_idx = np.where(np.logical_and(left_valid_int[:, i]>=left_T_pmin[i], left_valid_int[:, i]<=left_T_pmax[i]))
        diff_rgb_list.append(left_valid_rgb[diff_idx[0], i, :])
        diff_nml_list.append(left_valid_nml[diff_idx[0], i, :])
        inv_diff_idx = np.where(np.logical_or(left_valid_int[:, i]<left_T_pmin[i], left_valid_int[:, i]>left_T_pmax[i]))
        test_inv_diff_idx = inv_diff_idx[0].copy()
        test_inv_diff_idx = test_inv_diff_idx.reshape(1, -1).repeat(len(diff_idx[0]), 0)
        test_inv_diff_idx = np.abs(diff_idx[0].reshape(-1, 1) - test_inv_diff_idx)
        idx_of_diff_idx = np.argmin(test_inv_diff_idx, axis=0)
        diff_inv_idx = []
        for j in range(len(idx_of_diff_idx)):
            diff_inv_idx.append(diff_idx[0][idx_of_diff_idx[j]])
        diff_inv_rgb_list.append(left_valid_rgb[diff_inv_idx, i, :])
        diff_inv_nml_list.append(left_valid_nml[inv_diff_idx[0], i, :])

    diff_rgb_valid = torch.cat(diff_rgb_list, dim=0)
    diff_nml_valid = torch.cat(diff_nml_list, dim=0)
    diff_inv_rgb_valid = torch.cat(diff_inv_rgb_list, dim=0)
    diff_inv_nml_valid = torch.cat(diff_inv_nml_list, dim=0)
    
    diff_rgb_valid = torch.cat((diff_rgb_valid, diff_inv_rgb_valid), dim=0)
    diff_nml_valid = torch.cat((diff_nml_valid, diff_inv_nml_valid), dim=0)

    max_diff = diff_rgb_valid.mean(-1).max()
    min_diff = diff_rgb_valid.mean(-1).min()
    qut_diff = torch.quantile(diff_rgb_valid.mean(-1), 0.5)
    print(max_diff, min_diff, torch.quantile(diff_rgb_valid.mean(-1), 0.5))

    if "buddha" in cfg.data_path:
        max_diff = 1.0
        min_diff = 0.5
    else:
        if max_diff < 0.65:
            max_diff = 0.65
        if min_diff < 0.1 and qut_diff < 0.5 and qut_diff > 0.1:
            min_diff = torch.quantile(diff_rgb_valid.mean(-1), 0.5)
        elif min_diff < 0.1 and qut_diff < 0.1:
            min_diff = 0.2

    print(max_diff, min_diff)

    # ---------------------------------------------------------------------------------------
    # SH decomposition
    phi, theta = torch.meshgrid([torch.linspace(0., np.pi, PANO_H), 
                                 torch.linspace(-np.pi, np.pi, PANO_W)], indexing='ij')
    view_dirs = torch.stack([torch.sin(theta) * torch.sin(phi),
                             torch.cos(phi),
                             -torch.cos(theta) * torch.sin(phi)], dim=-1)             # [H, W, 3]
    ibl = fill_hole_with_nearest(view_dirs.view(-1, 3), diff_nml_valid, diff_rgb_valid).numpy().reshape(PANO_H, PANO_W, 3)

    xres = ibl.shape[1]
    sh_basis_matrix = getCoefficientsMatrix(xres, lmax=L_MAX)
    solidAngles = getSolidAngleMap(xres)
    nCoeffs = shTerms(L_MAX)
    iblCoeffs = np.zeros((nCoeffs,3))
    for i in range(0,shTerms(L_MAX)):
        iblCoeffs[i,0] = np.sum(ibl[:,:,0]*sh_basis_matrix[:,:,i]*solidAngles)
        iblCoeffs[i,1] = np.sum(ibl[:,:,1]*sh_basis_matrix[:,:,i]*solidAngles)
        iblCoeffs[i,2] = np.sum(ibl[:,:,2]*sh_basis_matrix[:,:,i]*solidAngles)
    sh_L = shReconstructSignal(iblCoeffs, width=xres).clip(0.)
    z_max = (diff_nml_valid[..., 1].reshape(-1)).max().numpy()
    z_min = (diff_nml_valid[..., 1].reshape(-1)).min().numpy()
    ck_view_dirs = view_dirs.reshape((PANO_H, PANO_W, 3))[:, 0, :]
    ck_view_dirs = ck_view_dirs[..., 1].numpy()
    ck_view_inds_max = np.concatenate(np.where((z_min*np.ones_like(ck_view_dirs)<ck_view_dirs))).max()
    ck_view_inds_min = np.concatenate(np.where((z_max*np.ones_like(ck_view_dirs)>ck_view_dirs))).min()

    view_dirs = view_dirs.reshape((PANO_H, PANO_W, 3))
    diff_rgb_valid = torch.from_numpy(sh_L)
    diff_rgb_valid = diff_rgb_valid.mean(-1)
    diff_rgb_valid = ((diff_rgb_valid - diff_rgb_valid.min())  / (diff_rgb_valid.max() - diff_rgb_valid.min())) * (max_diff - min_diff) + min_diff
    diff_nml_valid = view_dirs
    print(diff_rgb_valid.max(), diff_rgb_valid.min(), torch.quantile(diff_rgb_valid, 0.5))
    # ---------------------------------------------------------------------------------------
    out_dir = cfg.data_path
    numLgtSGs = cfg.num_SG
    
    init_dict = safe_init(numLgtSGs, "uniform")
    diff_rgb_valid = diff_rgb_valid.to(device).float()
    diff_nml_valid = diff_nml_valid.to(device).float()

    N_iter = cfg.N_iter

    lgtSG_lobes = torch.tensor(init_dict["lobe"]).to(device)
    lgtSG_mu  = torch.from_numpy(init_dict["mu"][..., 0:1]).to(device)
    lgtSG_lbd = torch.from_numpy(init_dict["lbd"]).to(device)
    lgtSGs = torch.cat((lgtSG_lobes, lgtSG_mu), dim=-1)
    lgtSGs.requires_grad = True

    optimizer = torch.optim.Adam([lgtSGs,], lr=1e-2)

    with trange(N_iter) as pbar:
        for step in pbar:
            optimizer.zero_grad()
            lgtSG_t = torch.cat((lgtSGs[..., :3], lgtSG_lbd, lgtSGs[..., 3:]), dim=-1)
            env_map = SG2Envmap(lgtSG_t, PANO_H, PANO_W)
            loss = torch.mean((env_map[ck_view_inds_min:ck_view_inds_max] - 
                               diff_rgb_valid[ck_view_inds_min:ck_view_inds_max]) ** 2)
            loss.backward()
            optimizer.step()

            pbar.set_description(f"loss:{loss.item():.8f}")

            if step % 100 == 0:
                envmap_check = env_map[ck_view_inds_min:ck_view_inds_max].clone().detach().cpu().numpy()
                gt_envmap_check = diff_rgb_valid[ck_view_inds_min:ck_view_inds_max].clone().detach().cpu().numpy()
                im = np.concatenate((gt_envmap_check, envmap_check), axis=0)
                
                im = np.clip(im, 0., 1.)
                im = np.uint8(im * 255.)

                lgtSG_rgb = torch.zeros((numLgtSGs, 7)).to(device)
                lgtSG_rgb[..., :3] = lgtSG_t[..., :3]
                lgtSG_rgb[..., 3:4]= lgtSG_t[..., 3:4]
                lgtSG_rgb[..., 4:] = lgtSG_t[..., 4:]
                
                out_img = np.concatenate((envmap_check[..., None].repeat(3, -1), gt_img.cpu().numpy(), ibl / ibl.max(), sh_L / sh_L.max(), sh_L.mean(-1)[..., None].repeat(3, -1) / sh_L.mean(-1)[..., None].max()), axis=0)
                np.save(os.path.join(out_dir, f'sg_{numLgtSGs}_{L_MAX}_{min_thresh}_{max_thresh}_{erode_mask}.npy'), lgtSG_rgb.clone().detach().cpu().numpy())
                cv2.imwrite(os.path.join(out_dir, f'sg_{numLgtSGs}_{L_MAX}_{min_thresh}_{max_thresh}_{erode_mask}.png'), out_img * 255.)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str,
        default="PATH"
    )
    parser.add_argument(
        "--num_SG", type=int,
        default=64
    )
    parser.add_argument(
        "--N_iter", type=int,
        default=10000
    )
    parser.add_argument(
        "--gpu", type=int,
        default=0
    )
    
    seed_everything(42)
    config = parser.parse_args()
    fit_env_map(cfg = config)