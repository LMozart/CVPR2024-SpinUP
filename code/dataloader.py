import torch
from torch.utils.data import Dataset
import numpy as np
import cv2 as cv
import torch.nn.functional as F
import scipy.io as scio
import random
import matplotlib.pyplot as plt


def lift(x, y, z, intrinsics):
    ''' Project a point from image space to camera space (for Tensor object) '''
    # For Tensor object
    # parse intrinsics
    fx = intrinsics[:, 0, 0].unsqueeze(1)                                           # [N]
    fy = intrinsics[:, 1, 1].unsqueeze(1)                                           # [N]
    cx = intrinsics[:, 0, 2].unsqueeze(1)                                           # [N]
    cy = intrinsics[:, 1, 2].unsqueeze(1)                                           # [N]
    sk = intrinsics[:, 0, 1].unsqueeze(1)                                           # [N]

    x_lift = (x - cx + cy * sk / fy - sk * y / fy) / fx * z                         # [N]
    y_lift = (y - cy) / fy * z                                                      # [N]

    # homogeneous coordinate
    return torch.stack([x_lift, y_lift, z, torch.ones_like(z)], dim=1)       # [N, 4, 1]


class Data_Loader(Dataset):
    def __init__(self, data_dict, gray_scale=False, data_len=1, mode='training', shadow_threshold=0.0, log_path="./"):
        # Camera Params
        cam_params = data_dict["cam_params"]
        self.K     = cam_params["K"]
        self.euler = cam_params["R"]
        self.focal = cam_params["f"]
        self.fs    = cam_params["fs"]

        self.gt_rot= data_dict["gt_rot"]
        
        # [Full Resolution Parts]
        # Images.
        self.images = torch.tensor(data_dict['images'], dtype=torch.float32)  # (num_images, height, width, channel)
        self.num_images = self.images.size(0)
        self.height     = self.images.size(1)
        self.width      = self.images.size(2)
        # Mask.
        self.mask = torch.tensor(data_dict['mask'], dtype=torch.float32)
        masks = self.mask[None,...].repeat((self.num_images,1,1))  # (num_images, height, width)
        self.valid_idx = torch.where(masks > 0.5)
        temp_idx       = torch.where(self.mask > 0.5)
        self.idx       = temp_idx
        # Normal.
        self.gt_normal = torch.tensor(data_dict['gt_normal'], dtype=torch.float32)
        self.gt_normal = F.normalize(self.gt_normal, dim=-1, p=2)
        # Contour Normal.
        self.pre_contour_normal = self.compute_contour_normal(data_dict['mask'])
        self.pre_contour_normal[..., 1:] = -self.pre_contour_normal[..., 1:]
        # Shadow.
        valid_shadow = self.update_valid_shadow_map(thres=shadow_threshold)
        self.shadow_imgs = torch.zeros_like(self.images)[..., 0]
        self.shadow_imgs[:, temp_idx[0], temp_idx[1]] = valid_shadow
        shadows = (self.shadow_imgs.view(-1, self.shadow_imgs.shape[-1]).numpy() * 255).astype(np.uint8)
        cv.imwrite(f"{log_path}/shadow_visual.png", shadows)
        # Configs.
        self.data_len = min(data_len, self.num_images)
        self.mode = mode
        # Cord XY.
        self.o_mask = self._get_outer_contour(self.mask.numpy())
        self.o_mask = torch.from_numpy(self.o_mask)
        self.o_idx  = torch.where(self.o_mask > 0.5)
        grid = torch.meshgrid(torch.arange(0, self.height), 
                              torch.arange(0, self.width))
        self.cord_idx = torch.stack((grid[0], grid[1]), dim=-1)
        self.cord_xy  = torch.stack((grid[1] / self.width, 
                                     grid[0] / self.height), dim=-1)
        # self.cord_xy  = (self.cord_xy - 0.5) * 2
        mean_cord  = self.cord_xy[self.o_idx].mean(0, keepdim=True)
        print(self.cord_xy.shape)
        # self.cord_xy_code  = self.cord_xy - mean_cord[None, ...]
        self.cord_xy_code  = (self.cord_xy - 0.5) * 2
        self.cord_xy  = (self.cord_xy - 0.5) * 2
        
        # View dirs.
        _, self.view_map = self._get_view(self.height, self.width)
        # Boundary.
        cnt_idxp = self.get_contour_idx()
        
        persp_normal = self._persp_contour_normal(self.view_map, self.pre_contour_normal, cnt_idxp)
        self.contour_normal = torch.zeros_like(self.gt_normal)
        self.contour_normal[cnt_idxp] = persp_normal
        test_gt_contour_normal = self.gt_normal[cnt_idxp]

        dot_product = (test_gt_contour_normal * persp_normal).sum(-1).clamp(-1, 1)
        angular_err = torch.acos(dot_product) * 180.0 / np.pi
        print(angular_err.mean())
        
        cnt_mask = torch.zeros_like(self.mask)
        cnt_mask[cnt_idxp] = 1.
        # Light
        self.light_maps = data_dict['lgt']
        self.lgt_SG     = data_dict["lgtSG"]

        # [Scaled Resolution Parts]
        self.scale = 4
        self.rand_p_idx  = 0
        self.random_size = self.scale ** 2

        kh, kw = self.scale, self.scale  # kernel size
        dh, dw = self.scale, self.scale  # stride
        self.p_images = self.images.unfold(1, kh, dh).unfold(2, kw, dw)
        self.p_mask   = self.mask.unfold(0, kh, dh).unfold(1, kw, dw)
        self.p_o_mask = self.o_mask.unfold(0, kh, dh).unfold(1, kw, dw)
        self.p_gt_nml = self.gt_normal.unfold(0, kh, dh).unfold(1, kw, dw)
        self.p_cord_xy= self.cord_xy.unfold(0, kh, dh).unfold(1, kw, dw)
        self.p_cord_xy_code= self.cord_xy_code.unfold(0, kh, dh).unfold(1, kw, dw)
        self.p_cnt_nml= self.contour_normal.unfold(0, kh, dh).unfold(1, kw, dw)
        self.p_cnt_msk= cnt_mask.unfold(0, kh, dh).unfold(1, kw, dw)
        self.p_view   = self.view_map.unfold(0, kh, dh).unfold(1, kw, dw)
        self.p_shadow = self.shadow_imgs.unfold(1, kh, dh).unfold(2, kw, dw)
        self.p_cord_idx= self.cord_idx.unfold(0, kh, dh).unfold(1, kw, dw)


        p_im_xsz, p_im_ysz = self.p_images.shape[1], self.p_images.shape[2]
        self.p_images = self.p_images.reshape(self.num_images, p_im_xsz, p_im_ysz, 3, -1)
        self.p_mask   = self.p_mask.reshape(p_im_xsz, p_im_ysz, -1)
        self.p_o_mask = self.p_o_mask.reshape(p_im_xsz, p_im_ysz, -1)
        self.p_gt_nml = self.p_gt_nml.reshape(p_im_xsz, p_im_ysz, 3, -1)
        self.p_cord_xy= self.p_cord_xy.reshape(p_im_xsz, p_im_ysz, 2, -1)
        self.p_cord_xy_code= self.p_cord_xy_code.reshape(p_im_xsz, p_im_ysz, 2, -1)
        self.p_cnt_nml= self.p_cnt_nml.reshape(p_im_xsz, p_im_ysz, 3, -1)
        self.p_cnt_msk= self.p_cnt_msk.reshape(p_im_xsz, p_im_ysz, -1)
        self.p_view   = self.p_view.reshape(p_im_xsz, p_im_ysz, 3, -1)
        self.p_shadow = self.p_shadow.reshape(self.num_images, p_im_xsz, p_im_ysz, -1)
        self.p_cord_idx= self.p_cord_idx.reshape(p_im_xsz, p_im_ysz, 2, -1)

        self.mm_p = 4 + 1
        self.get_idxp_and_mask()

    def __len__(self):
        if self.mode == 'testing' or self.mode == 'valid':
            return self.data_len
        else:
            raise NotImplementedError('Dataloader mode unknown')

    def __getitem__(self, idx):
        if self.mode == 'testing':
            return self.get_testing_rays(idx)
        elif self.mode == 'valid':
            return self.get_full_res_rays(idx)
        
    def _get_outer_contour(self, mask):
        dilation = cv.dilate(mask, np.ones((3, 3)), iterations = 1)
        return dilation
    
    def get_pretrained_nml_data(self):
        input_data_dict = {"uv":      self.valid_ocord,
                           "fit_nml": self.fit_nml}
        return input_data_dict
    
    def apply_fpe(self, filter, encoder):
        self.cord_xy = encoder(self.cord_xy)
        self.cord_xy = self.cord_xy[None, ...].permute(0, 3, 1, 2)    
        self.cord_xy = filter(self.cord_xy)[0].permute(1, 2, 0)
        print("Applying FPE in Dataloader:", self.cord_xy.shape)
    
    def resample_patch_idx(self):
        self.rand_p_idx = random.randrange(self.random_size)
    
    def shrink_scale(self):
        if self.mm_p != 1:
            self.mm_p = self.mm_p - 2
            if self.mm_p < 1:
                self.mm_p = 1
        print(f"Shrinking Scale:", self.mm_p)

    def get_idxp_and_mask(self, mode="valid"):
        if self.mode == "testing":
            mask     = self.p_mask[..., self.rand_p_idx]
            res = {"mask": mask}
        else:
            mask     = self.mask
            bbox_int = self.get_bounding_box_int(mask)
            res = {"mask":     mask,
                   "bbox_int": bbox_int}
        self.idxp= torch.where(mask > 0.)
        res["idxp"] = self.idxp
        return res
    
    def _persp_contour_normal(self, view_map, cnt_nml, idxp_contour):
        v_xyz = view_map[idxp_contour]
        theta = torch.atan2(cnt_nml[idxp_contour[0], idxp_contour[1], 1], cnt_nml[idxp_contour[0], idxp_contour[1], 0])    
        phi   = torch.atan(-v_xyz[..., 2] / (v_xyz[..., 0] * torch.cos(theta) + v_xyz[..., 1] * torch.sin(theta)))
        phi[phi < 0] = np.pi + phi[phi < 0]

        cnt_x = torch.cos(theta) * torch.sin(phi)
        cnt_y = torch.sin(theta) * torch.sin(phi)
        cnt_z = torch.cos(phi)
        cnt_nmls = torch.stack((cnt_x, 
                                cnt_y, 
                                cnt_z + 0.1), dim=-1)
        cnt_nmls = F.normalize(cnt_nmls, p=2, dim=-1)
        return cnt_nmls
    
    def _get_view(self, H, W):
        uv = np.mgrid[:H, :W]
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)                                             # [HW, 2]

        K = torch.from_numpy(np.array(self.K)).float()

        xyz = torch.cat((uv, torch.ones_like(uv[..., 0:1])), dim=-1).float()
        xyz = lift(xyz[..., 0, None], xyz[..., 1, None], xyz[..., 2, None], K[None, ...])[..., :3, 0]
        xyz[..., 0] = -xyz[..., 0]
        xyz = xyz.reshape(H, W, 3)

        view_dirs = xyz[self.idx]
        view_dirs = F.normalize(view_dirs, p=2, dim=-1)
        xyz = F.normalize(xyz, p=2, dim=-1)
        return view_dirs, xyz

    def get_full_res_rays(self, ith):
        mm_size = 5
        mm_x = torch.tensor([[0,  0, -1, 0, 1]])
        mm_y = torch.tensor([[0, -1,  0, 1, 0]])

        cord_idx   = self.cord_idx[self.idxp]
        cord_idx_x = cord_idx[..., 0]
        cord_idx_y = cord_idx[..., 1]
        
        p_idx_x = cord_idx_x.reshape(1, -1).repeat(mm_size, 1) + mm_x.reshape(-1, 1)
        p_idx_y = cord_idx_y.reshape(1, -1).repeat(mm_size, 1) + mm_y.reshape(-1, 1)

        p_idx_xy = (p_idx_x.long(), 
                    p_idx_y.long())
        
        # Valid XY
        valid_xy_code = self.cord_xy_code[p_idx_xy]
        valid_xy = self.cord_xy[p_idx_xy]

        # Valid RGB
        valid_rgb = self.images[ith][self.idxp]
        # Valid Normal
        valid_nml = self.gt_normal[self.idxp]
        # Valid View
        valid_view= self.view_map[self.idxp]
        # Light
        lgt_map  = self.light_maps[ith]
        
        sample = {'item_idx': ith,
                  'valid_xy': valid_xy,
                  'rgb'     : valid_rgb,
                  'normal'  : valid_nml,
                  'view'    : valid_view,
                  'lgt'     : lgt_map,
                  "p_idx_xy": p_idx_xy,
                  "valid_xy_code": valid_xy_code}
        
        sample['shadow_mask']    = self.shadow_imgs[ith][self.idxp]
        sample['contour_normal'] = self.contour_normal[self.idxp]
        return sample

    def get_testing_rays(self, ith):  
        mm_x = torch.tensor([[0,  0, -self.mm_p, 0, self.mm_p]])
        mm_y = torch.tensor([[0, -self.mm_p,  0, self.mm_p, 0]])

        # 
        cord_idx = self.p_cord_idx[..., self.rand_p_idx]
        cord_idx = cord_idx[self.idxp]
        cord_idx_x = cord_idx[..., 0]
        cord_idx_y = cord_idx[..., 1]
        
        p_idx_x = cord_idx_x.reshape(1, -1).repeat(5, 1) + mm_x.reshape(-1, 1)
        p_idx_y = cord_idx_y.reshape(1, -1).repeat(5, 1) + mm_y.reshape(-1, 1)

        p_idx_xy = (p_idx_x.long(), 
                    p_idx_y.long())
        
        # Valid XY        
        # valid_xy_test = self.cord_xy[p_idx_xy]
        valid_xy_code = self.cord_xy_code[p_idx_xy]
        valid_xy = self.cord_xy[p_idx_xy]

        # Valid RGB
        valid_rgb = self.p_images[ith][..., self.rand_p_idx][self.idxp]
        # Valid Normal
        valid_nml = self.p_gt_nml[..., self.rand_p_idx][self.idxp]
        # Valid View
        valid_view= self.p_view[..., self.rand_p_idx][self.idxp]
        # Light
        lgt_map  = self.light_maps[ith]
        
        sample = {'item_idx': ith,
                  'rgb'     : valid_rgb,
                  'normal'  : valid_nml,
                  'view'    : valid_view,
                  'lgt'     : lgt_map,
                  "p_idx_xy": p_idx_xy,
                  "valid_xy": valid_xy,
                  "valid_xy_code": valid_xy_code,
                  "img_mean": self.p_images.mean(0)[..., self.rand_p_idx],
                  "gt_rot":   self.gt_rot[ith]}
        
        sample['shadow_mask']    = self.p_shadow[ith][..., self.rand_p_idx][self.idxp]
        sample['contour_normal'] = self.p_cnt_nml[..., self.rand_p_idx][self.idxp]
        return sample

    def get_mask(self):
        return self.mask

    def get_bounding_box(self):
        return self.valid_input_iwih_max, self.valid_input_iwih_min

    def get_bounding_box_int(self, mask):
        mask = mask.numpy()
        valididx = np.where(mask > 0.5)
        xmin = valididx[0].min()
        xmax = valididx[0].max()
        ymin = valididx[1].min()
        ymax = valididx[1].max()

        xmin = max(0, xmin - 1)
        xmax = min(xmax + 2, mask.shape[0])
        ymin = max(0, ymin - 1)
        ymax = min(ymax + 2, mask.shape[1])
        return xmin, xmax, ymin, ymax

    def get_all_light_encoding(self):
        return self.ld_encoding

    def get_all_masked_images(self):
        idx = torch.where(self.mask > 0.5)
        x_max, x_min = max(idx[0]), min(idx[0])
        y_max, y_min = max(idx[1]), min(idx[1])

        x_max, x_min = min(x_max+15, self.images.shape[1]), max(x_min-15, 0)
        y_max, y_min = min(y_max+15, self.images.shape[2]), max(y_min-15, 0)

        out_images = self.images[:, x_min:x_max, y_min:y_max, :].permute([0,3,1,2])
        out_masks = self.mask[x_min:x_max, y_min:y_max][None, None, ...].repeat(out_images.size(0),1,1,1)
        out = torch.cat([out_images, out_masks], dim=1)
        return out  # (num_image, 4, height, width)

    def get_contour_idx(self):
        mask_x1, mask_x2, mask_y1, mask_y2 = self.mask.clone(), self.mask.clone(), self.mask.clone(), self.mask.clone()
        mask_x1[:-1, :] = self.mask[1:, :]
        mask_x2[1:, :] = self.mask[:-1, :]
        mask_y1[:, :-1] = self.mask[:, 1:]
        mask_y2[:, 1:] = self.mask[:, :-1]
        mask_1 = mask_x1 * mask_x2 * mask_y1 * mask_y2
        idxp_contour = torch.where((mask_1 < 0.5) & (self.mask > 0.5))

        contour_map = torch.zeros_like(self.mask)
        contour_map[idxp_contour] = 1

        self.contour = contour_map[torch.where(self.mask>0.5)]
        self.contour_normal = self.contour[:,None] * self.pre_contour_normal[torch.where(self.mask>0.5)]
        return idxp_contour

    def update_valid_shadow_map(self, thres):
        valid_rgb = self.images[self.valid_idx]
        valid_rgb = valid_rgb.view(self.num_images, -1, 3)
        if valid_rgb.size(-1) == 3:
            temp_rgb = valid_rgb.mean(dim=-1)  # (num_image, num_mask_point)
        temp_rgb_topk_mean = torch.topk(temp_rgb, k=int(len(temp_rgb)*0.9), dim=0, largest=False)[0].mean(dim=0, keepdim=True)
        idxp = torch.where(0.8*temp_rgb_topk_mean <= temp_rgb)
        valid_shadow = torch.zeros_like(temp_rgb)
        valid_shadow = 0. * temp_rgb / (0.8 * temp_rgb_topk_mean)
        valid_shadow[idxp] = 1
        return valid_shadow

    @staticmethod
    def compute_contour_normal(_mask):
        blur = cv.GaussianBlur(_mask, (11, 11), 0)
        n_x = -cv.Sobel(blur, cv.CV_32F, 1, 0, ksize=11, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        n_y = -cv.Sobel(blur, cv.CV_32F, 0, 1, ksize=11, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)

        n = np.sqrt(n_x**2 + n_y**2) + 1e-5
        contour_normal = np.zeros((_mask.shape[0], _mask.shape[1], 3), np.float32)
        contour_normal[:, :, 0] = n_x / n
        contour_normal[:, :, 1] = n_y / n
        return torch.tensor(contour_normal, dtype=torch.float32)