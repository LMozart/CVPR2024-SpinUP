import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from math_utils import saturate
from utils import normalize, dot, axis_angle_rotation

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

class Light_Model_SG(nn.Module):
    def __init__(self, num_sgs, device, SG_init, euler_init, num_imgs=20, load_pretrained=True, trainable=True):
        super(Light_Model_SG, self).__init__()
        self.device = device
        self.N_l    = num_sgs

        if load_pretrained:
            lgtSG = SG_init
            self.lgtSG_lobe = torch.from_numpy(lgtSG[..., :3]).to(device).float()
            self.lgtSG_lbd  = torch.from_numpy(np.ones_like(lgtSG[..., 3:4]) * 0.65).to(device).float()
            self.lgtSG_lbd  = inverse_sigmoid(self.lgtSG_lbd)
            self.lgtSG_mu   = torch.from_numpy(lgtSG[..., -3:]).to(device).float()
            print("mu:", self.lgtSG_mu.max())
            print("Loading ...")
        else:
            lgt_dict = self.safe_init(num_sgs=num_sgs, mode="uniform")
            self.lgtSG_mu   = torch.from_numpy(lgt_dict["mu"]).to(device).float()      # [N_l, 3]
            self.lgtSG_lbd  = torch.from_numpy(lgt_dict["lbd"]).to(device).float()     # [N_l, 1]
            self.lgtSG_lobe = torch.from_numpy(lgt_dict["lobe"]).to(device).float()    # [N_l, 3]
            self.lgtSG_lobe = F.normalize(self.lgtSG_lobe)
            
        self.lgtSG_mu   = nn.Parameter(self.lgtSG_mu[None, ...], requires_grad=trainable)   # [1, N_l, 3]
        self.lgtSG_lbd  = nn.Parameter(self.lgtSG_lbd[None, ...], requires_grad=trainable)  # [1, N_l, 1]
        self.lgtSG_lobe = nn.Parameter(self.lgtSG_lobe[None, ...], requires_grad=trainable) # [1, N_l, 3]

        # Euler for rotation.
        euler = torch.from_numpy(euler_init).float()
        euler = euler.view(num_imgs, 1)
        self.euler = nn.Parameter(euler, requires_grad=False)

    def forward(self, idx, num_rays):
        N_bs = len(idx)

        # Prepare for the light direction's rotation.
        # ---------------------------------------------------------------------------
        rots_euler = self.euler
        # ---------------------------------------------------------------------------

        rot_mats = []
        rot_mats = axis_angle_rotation("Y", rots_euler[idx])[:, 0]
        rot_mats = rot_mats[:, None, ...].repeat(1, num_rays//N_bs, 1, 1)        # [N_bs, N_pts, 3, 3])
        rot_mats = rot_mats.view(num_rays, 3, 3)                                 # [N, 3, 3]
        rot_mats = rot_mats.to(self.device)                                      # [N, 3, 3]
        
        # Repeat for batch size.
        lgtSG_lobe = self.lgtSG_lobe.repeat(num_rays, 1, 1) # [N, N_l, 3]
        lgtSG_lobe = lgtSG_lobe[..., None, :] * rot_mats[:, None, ...] # [N, N_l, 3, 3]
        lgtSG_lobe = lgtSG_lobe.sum(-1)                     # [N, N_l, 3]
        lgtSG_lbd  = self.lgtSG_lbd.repeat(num_rays, 1, 1)  # [N, N_l, 1]
        lgtSG_mu   = self.lgtSG_mu.repeat(num_rays, 1, 1)   # [N, N_l, 3]

        # Preprocessing of the SG parameters
        lgtSG_lobe = F.normalize(lgtSG_lobe, p=2, dim=-1)                   # [N, N_l, 3]
        lgtSG_lbd  = torch.log(torch.sigmoid(lgtSG_lbd)) * -74.4891       # [N, N_l, 3]
        lgtSG_mu   = torch.abs(lgtSG_mu)                                    # [N, N_l, 3]
        lgtSG      = torch.cat((lgtSG_lobe, lgtSG_lbd, lgtSG_mu), dim=-1)   # [N, N_l, 7]

        res_dict = {"lgtSG": lgtSG,   
                    "rot_mats": rot_mats} 

        return res_dict

    def safe_init(self, num_sgs, mode="uniform"):
        if mode=="uniform":
            axis = []
            inc = np.pi * (3.0 - np.sqrt(5.0))
            off = 2.0 / num_sgs
            for k in range(num_sgs):
                y = k * off - 1.0 + (off / 2.0)
                r = np.sqrt(1.0 - y * y)
                phi = k * inc
                axis.append(normalize(np.array([np.sin(phi) * r, y, -np.cos(phi) * r])))

            minDp = 1.0
            for a in axis:
                h = normalize(a + axis[0])
                minDp = min(minDp, dot(h, axis[0]))

            sharpness = (np.log(0.65) * num_sgs) / (minDp - 1.0)

            lobe = np.stack(axis, 0)  # Shape: num_sgs, 3
            mu   = np.ones_like(axis)
            lbd  = np.ones((num_sgs, 1)) * sharpness
        elif mode=="random":
            lobe = np.random.randn(num_sgs, 3)      # [N_l, 3]
            mu   = np.random.randn(num_sgs, 3)       # [N_l, 3]
            lbd  = np.random.randn(num_sgs, 1) # [N_l, 3]
            lbd = lbd * 100.
            lobe = normalize(lobe)                             # [N_l, 3]
        
        res_dict = {"lobe": lobe,
                    "mu": mu,
                    "lbd": lbd}
        return res_dict

    def get_light_from_idx(self, idx, num_rays):
        lgtSG = self.forward(idx, num_rays)
        return lgtSG
    
    def set_trainable_true(self):
        self.euler.requires_grad = True


# Implementation borrowed from: https://github.com/krrish94/nerf-pytorch
class NormalMLP(torch.nn.Module):
    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=3,
        encode_fn=None,          # Encoder   
        num_encoding_fn_input=10,# Encode Dimension
        include_input_input=2,    # denote images coordinates (u, v)
    ):
        super(NormalMLP, self).__init__()
        self.dim_uv = include_input_input * (1 + 2 * num_encoding_fn_input)
        self.skip_connect_every = skip_connect_every + 1
        self.encode_fn = encode_fn

        # Branch 1
        self.layers_depth = torch.nn.ModuleList()
        self.layers_depth.append(torch.nn.Linear(self.dim_uv, hidden_size))
        for i in range(1, num_layers):
            if i == self.skip_connect_every:
                self.layers_depth.append(torch.nn.Linear(self.dim_uv + hidden_size, hidden_size))
            else:
                self.layers_depth.append(torch.nn.Linear(hidden_size, hidden_size))

        # Branch Output
        self.fc_z = torch.nn.Linear(hidden_size, 1)

        # Activation Function
        self.relu = torch.nn.functional.relu

    def forward(self, input):
        # Inputs.
        input = self.encode_fn(input)

        # Compute Features.
        xyz = input[..., :self.dim_uv]
        x = xyz
        for i in range(len(self.layers_depth)):
            if i == self.skip_connect_every:
                x = self.layers_depth[i](torch.cat((xyz, x), -1))
            else:
                x = self.layers_depth[i](x)
            x = self.relu(x)
        
        # Predict Depth.
        nml_z = torch.abs(self.fc_z(x))
        return nml_z
    
    def depth_forward(self, x):
        return self.forward(x)
    
    def gradient(self, x):
        x.requires_grad_(True)
        y = self.depth_forward(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)
        

class MaterialMLP(torch.nn.Module):
    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=3,
        encode_fn=None,            # Encoder
        num_encoding_fn_input=10,  # Encode Dimension
        include_input_input=2,     # Images Coordinates (u, v)
        output_ch=1
    ):
        super(MaterialMLP, self).__init__()
        self.dim_uv = include_input_input * (1 + 2 * num_encoding_fn_input)
        self.skip_connect_every = skip_connect_every + 1
        self.encode_fn = encode_fn

        # Branch 1
        self.layers_mat = torch.nn.ModuleList()
        self.layers_mat.append(torch.nn.Linear(self.dim_uv, hidden_size))
        for i in range(1, num_layers):
            if i == self.skip_connect_every:
                self.layers_mat.append(torch.nn.Linear(self.dim_uv + hidden_size, hidden_size))
            else:
                self.layers_mat.append(torch.nn.Linear(hidden_size, hidden_size))                            
        self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        
        # Branch 2
        self.layers_coeff = torch.nn.ModuleList()
        self.layers_coeff.append(torch.nn.Linear(hidden_size, hidden_size // 2))
        for i in range(3):
            self.layers_coeff.append(torch.nn.Linear(hidden_size // 2, hidden_size // 2))
        
        # Branch Output
        self.fc_spec_coeff = torch.nn.Linear(hidden_size // 2, output_ch)
        self.fc_diff = torch.nn.Linear(hidden_size // 2, 3)

        # Activation Function
        self.relu = torch.nn.functional.relu

    def forward(self, input):
        # Inputs.
        input = self.encode_fn(input)
        uv = input[..., :self.dim_uv]
        x  = uv

        # Compute Features.
        for i in range(len(self.layers_mat)):
            if i == self.skip_connect_every:
                x = self.layers_mat[i](torch.cat((uv, x), -1))
            else:
                x = self.layers_mat[i](x)
            x = self.relu(x)
        feat = self.fc_feat(x)
        
        # Predict Coefficients.
        x = self.layers_coeff[0](feat)
        x = self.relu(x)
        for i in range(1, len(self.layers_coeff)):
            x = self.layers_coeff[i](x)
            x = self.relu(x)
            
        diff       = torch.abs(self.fc_diff(x))
        spec_coeff = torch.abs(self.fc_spec_coeff(x))
        
        # Output Recording.
        coeff_dict = {"diff": diff, 
                      "spec_coeff": spec_coeff}
        return coeff_dict

    def mat_forward(self, x):
        return self.forward(x)
    
    def gradient(self, x):
        x.requires_grad_(True)
        y = self.mat_forward(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


class SpinLightMLP(torch.nn.Module):
    def __init__(
        self,
        num_layers=8,
        hidden_size=256,
        skip_connect_every=3,
        encode_fn=None,          # Encoder   
        num_encoding_fn_input=10,# Encode Dimension
        include_input_input=2,    # denote images coordinates (u, v)
        output_ch=1
    ):
        super(SpinLightMLP, self).__init__()
        self.dim_uv = include_input_input * (1 + 2 * num_encoding_fn_input)
        self.skip_connect_every = skip_connect_every + 1
        self.encode_fn = encode_fn

        # Branch Encoder
        self.layers_encode = torch.nn.ModuleList()
        self.layers_encode.append(torch.nn.Linear(self.dim_uv, hidden_size))
        for i in range(1, num_layers):
            if i == self.skip_connect_every:
                self.layers_encode.append(torch.nn.Linear(self.dim_uv + hidden_size, hidden_size))
            else:
                self.layers_encode.append(torch.nn.Linear(hidden_size, hidden_size))

        # Branch Depth
        self.fc_d_feat = torch.nn.Linear(hidden_size, hidden_size)
        self.layers_depth = torch.nn.ModuleList()
        self.layers_depth.append(torch.nn.Linear(hidden_size, hidden_size // 2))
        for i in range(3):
            self.layers_depth.append(torch.nn.Linear(hidden_size // 2, hidden_size // 2))
        self.fc_z = torch.nn.Linear(hidden_size // 2, 1)

        # Branch Mat
        self.fc_feat = torch.nn.Linear(hidden_size, hidden_size)
        
        self.layers_coeff = torch.nn.ModuleList()
        self.layers_coeff.append(torch.nn.Linear(hidden_size, hidden_size // 2))
        for i in range(3):
            self.layers_coeff.append(torch.nn.Linear(hidden_size // 2, hidden_size // 2))
        
        self.fc_spec_coeff = torch.nn.Linear(hidden_size // 2, output_ch)
        self.fc_diff = torch.nn.Linear(hidden_size // 2, 3)

        # Activation Function
        self.relu = torch.nn.functional.relu

    def forward(self, input):
        # Inputs.
        input = self.encode_fn(input)

        # Compute Features.
        xyz = input[..., :self.dim_uv]
        x = xyz
        for i in range(len(self.layers_encode)):
            if i == self.skip_connect_every:
                x = self.layers_encode[i](torch.cat((xyz, x), -1))
            else:
                x = self.layers_encode[i](x)
            x = self.relu(x)
        
        # Predict Depth.
        feat_depth = self.fc_d_feat(x)
        d_x = self.layers_depth[0](feat_depth)
        d_x = self.relu(d_x)
        for i in range(1, len(self.layers_depth)):
            d_x = self.layers_depth[i](d_x)
            d_x = self.relu(d_x)
        depth = torch.abs(self.fc_z(d_x))

        # Predict Mat.
        feat = self.fc_feat(x)
        
        # Predict Coefficients.
        x = self.layers_coeff[0](feat)
        x = self.relu(x)
        for i in range(1, len(self.layers_coeff)):
            x = self.layers_coeff[i](x)
            x = self.relu(x)
            
        diff       = torch.abs(self.fc_diff(x))
        spec_coeff = torch.abs(self.fc_spec_coeff(x))
        
        # Output Recording.
        output_dict = {"diff": diff, 
                       "spec_coeff": spec_coeff,
                       "depth": depth}
        return output_dict


# Code reimplemented from: https://github.com/junxuan-li/SCPS-NIR
class SG(nn.Module):
    def __init__(
            self,
            num_bases,
            k_low,
            k_high,
            trainable_k,
    ):
        super(SG, self).__init__()
        self.num_bases   = num_bases
        self.trainable_k = trainable_k
        kh = k_high
        kl = k_low
        self.k = nn.Parameter(torch.linspace(kl, kh, num_bases, dtype=torch.float32)[None, :], requires_grad=True)
        self.optimizer = optim.Adam([{'params': self.k}], lr=1e-3)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 1500, gamma = 0.1)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()

    def step_scheduler(self):
        self.scheduler.step()

    def forward(self, device, current_epoch):
        if self.trainable_k:
            k = self.k
        else:
            k = self.k.to(device)
        rate = torch.abs(k).clip(0.01, 1.50)
        
        if current_epoch == 250:
            replace_rate = rate.clone()
            replace_rate.require_grads = False
            if torch.abs(replace_rate[0, 0] - replace_rate[0, 1]) > 0.5:
                replace_rate[0, 0] = replace_rate[0, 1]
                self.k = nn.Parameter(replace_rate, requires_grad=True)
                self.optimizer = optim.Adam([{'params': self.k}], lr=1e-3)
        return rate

def dynamic_basis(input, current_epoch, total_epoch, num_bases):
    """
    Args:
        input:  (batch, num_bases, 3)
        current_epoch:
        total_epoch:
        num_bases:
    Returns:
    """
    alpha = current_epoch / total_epoch * (num_bases)
    k = torch.arange(num_bases, dtype=torch.float32, device=input.device)
    weight = (1 - (alpha - k).clamp_(min=0, max=1).mul_(math.pi).cos_()) / 2
    weight = weight[None, :, None]
    weighted_input = input * weight
    return weighted_input


def totalVariation(image, mask, num_rays):
    pixel_dif1 = torch.abs(image[1:, :, :] - image[:-1, :, :]).sum(dim=-1) * mask[1:, :] * mask[:-1, :]
    pixel_dif2 = torch.abs(image[:, 1:, :] - image[:, :-1, :]).sum(dim=-1) * mask[:, 1:] * mask[:, :-1]
    tot_var = (pixel_dif1.sum() + pixel_dif2.sum()) / num_rays
    return tot_var

def totalVariation_L2(image, mask, num_rays):
    pixel_dif1 = torch.square(image[1:, :, :] - image[:-1, :, :]).sum(dim=-1) * mask[1:, :] * mask[:-1, :]
    pixel_dif2 = torch.square(image[:, 1:, :] - image[:, :-1, :]).sum(dim=-1) * mask[:, 1:] * mask[:, :-1]
    tot_var = (pixel_dif1.sum() + pixel_dif2.sum()) / num_rays
    return tot_var
