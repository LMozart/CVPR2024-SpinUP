import torch
import numpy as np
import torch.nn.functional as F
from typing import Tuple
from math_utils import *

def calculate_sg_basis(view_dirs, SG_lobes, SG_lambdas, SG_mus):
    """Calculate SG basis. 
    Referenced to "All-frequency rendering of dynamic, spatially-varying reflectance".
    """
    return SG_mus * torch.exp(SG_lambdas * (dot(view_dirs, SG_lobes) - 1.))

def render_envmap(lgtSGs:torch.Tensor, 
                  H:int, W:int)->torch.Tensor:
    """Render Environment Map from Spherical Gaussians. 
    The code is reused from https://github.com/Kai-46/PhySG. Particularly, we assume the Environment Map is rendered in the same coordinates of normal map in Photometric Stereo, which is given below:
        y    
        |   
        |  
        | 
         --------->   x
       /
      /
    z 

    Args:
        lgtSGs:  torch.Tensor, [N_l, 7] array of Spherical Gaussians representing environment map.
        normals: torch.Tensor, [N_l, 3] array of normal directions (assumed to be unit vectors).
    Returns:
        rgb: torch.Tensor, [H, W, 3], array of reflection directions.
    """
    N_l = lgtSGs.shape[0]
    
    phi, theta = torch.meshgrid([torch.linspace(0., np.pi, 256), torch.linspace(-np.pi, np.pi, 512)], indexing='ij')
    view_dirs = torch.stack([torch.sin(theta) * torch.sin(phi),
                             torch.cos(phi),
                             -torch.cos(theta) * torch.sin(phi)], dim=-1)             # [H, W, 3]
    view_dirs = view_dirs.to(lgtSGs.device)                                           # [H, W, 3]
    view_dirs = view_dirs.unsqueeze(-2)                                               # [H, W, 1, 3]
    lgtSGs = lgtSGs.view(1, 1, N_l, 7).repeat(H, W, 1, 1)                             # [H, W, N_l, 3]

    lgtSG_lobes  = F.normalize(lgtSGs[..., :3], p=2, dim=-1)                          # [H, W, N_l, 3]
    lgtSG_lambdas= torch.abs(lgtSGs[..., 3:4])                                        # [H, W, N_l, 3]
    lgtSG_mus    = torch.abs(lgtSGs[..., -3:])                                        # [H, W, N_l, 3]
    rgb    = calculate_sg_basis(view_dirs, lgtSG_lobes, lgtSG_lambdas, lgtSG_mus)     # [H, W, N_l, 3]
    rgb    = torch.sum(rgb, dim=-2)                                                   # [H, W, 3]
    envmap = rgb.reshape((H, W, 3)).clip(min=0.)                                      # [H, W, 3]
    return envmap

# SG calculation.
def _extract_sg_components(sg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    """
    s_amplitude = sg[..., -3:]
    s_sharpness = sg[..., 3:4]
    s_axis      = sg[..., :3]
    return (
        torch.abs(s_amplitude),
        F.normalize(s_axis, p=2, dim=-1),
        torch.abs(s_sharpness),
    )

def _sg_integral(sg: torch.Tensor) -> torch.Tensor:
    s_amplitude, _, s_sharpness = _extract_sg_components(sg)
    expTerm = 1.0 - safe_exp(-2.0 * s_sharpness)
    return 2 * np.pi * safe_div(s_amplitude, s_sharpness)  * expTerm

def _sg_inner_product(sg1: torch.Tensor, sg2: torch.Tensor) -> torch.Tensor:
    s1_amplitude, s1_axis, s1_sharpness = _extract_sg_components(sg1)
    s2_amplitude, s2_axis, s2_sharpness = _extract_sg_components(sg2)

    umLength = magnitude(
        s1_sharpness * s1_axis + s2_sharpness * s2_axis
    )
    expo = (
        safe_exp(umLength - s1_sharpness - s2_sharpness)
        * s1_amplitude
        * s2_amplitude
    )
    other = 1.0 - safe_exp(-2.0 * umLength)
    return safe_div(2.0 * np.pi * expo * other, umLength)

def _stack_sg_components(axis, sharpness, amplitude):
    return torch.cat(
        [axis,
         sharpness,
         amplitude,],
        -1,
    )

# Warp specular.
def _warp_specular(ndf, v):
    ndf_amp, ndf_axis, ndf_sharpness = _extract_sg_components(ndf)
    warp_ndf_axis = reflect(-v, ndf_axis)
    warp_ndf_amp  = ndf_amp
    warp_ndf_sharpness = safe_div(ndf_sharpness, 4.0 * saturate(dot(ndf_axis, v), 1e-4))
    return torch.cat((warp_ndf_axis, warp_ndf_sharpness, warp_ndf_amp), dim=-1)

def _distribution_term(d: torch.Tensor, roughness: torch.Tensor) -> torch.Tensor:
    a2 = saturate(roughness * roughness, 1e-3)

    ret = _stack_sg_components(
        d,
        2.0 / torch.maximum(a2, torch.ones_like(a2) * 1e-6),
        torch.reciprocal(np.pi * a2).repeat(1, 1, 1, 3),
    )
    return ret

def _ggx(a2: torch.Tensor, ndx: torch.Tensor) -> torch.Tensor:
    return torch.reciprocal(ndx + safe_sqrt(a2 + (1 - a2) * ndx * ndx))

def evaluate_specular(v: torch.Tensor, 
                      base_color: torch.Tensor,
                      nml: torch.Tensor, 
                      weights,
                      roughness: torch.Tensor,
                      lgtSG: torch.Tensor):
    """ Parameters
    Args:
        v:    torch.Tensor, [N, 1, 3]
        nml:  torch.Tensor, [N, 1, 3]
        roughness: torch.Tensor, [N, 1, 1]
        metallic:  torch.Tensor, [N, 1, 1]
    Returns:
        rgb: torch.Tensor, [H, W, 3], array of reflection directions.
    """
    ndf = _distribution_term(nml, roughness)
    warp_ndf = _warp_specular(ndf[..., 0, :], v[..., 0, :])
    _, warpDir, _ = _extract_sg_components(warp_ndf)
    warpDir = warpDir[:, None]
    warp_ndf= warp_ndf[:, None]

    ndl = saturate(dot(nml, warpDir))
    ndv = saturate(dot(nml, v))
    h   = F.normalize(warpDir + v, p=2, dim=-1)
    ldh = saturate(dot(warpDir, h))
    powTerm  = torch.pow(1.0 - ldh, 5)
    specular = base_color
    
    a2  = saturate(roughness * roughness, 1e-3)
    D   = _sg_inner_product(warp_ndf, lgtSG)
    G   = _ggx(a2, ndl) * _ggx(a2, ndv)
    Fsl = specular + (1.0 - specular) * powTerm

    ret = D * G * Fsl * ndl * weights
    return ret.sum(-2).sum(-2)

def evaluate_diffuse(diffuse, sg_illuminations, normal):
    '''_summary_
    '''
    diff = diffuse / np.pi
    # diff = diffuse
    _, s_axis, s_sharpness = _extract_sg_components(sg_illuminations)
    mudn = saturate(dot(s_axis, normal))

    c0 = 0.36
    c1 = 1.0 / (4.0 * c0)

    eml  = safe_exp(-s_sharpness)
    em2l = eml * eml
    rl   = safe_reciprocal(s_sharpness)

    scale = 1.0 + 2.0 * em2l - rl
    bias = (eml - em2l) * rl - em2l

    x = safe_sqrt(1.0 - scale)
    x0 = c0 * mudn
    x1 = c1 * x

    n = x0 + x1

    y_cond = torch.le(torch.abs(x0), x1)
    y_true = n * (n / torch.maximum(x, torch.ones_like(x) * 1e-6))
    y_false = mudn
    y = torch.where(y_cond, y_true, y_false)

    res = scale * y + bias
    res = res * _sg_integral(sg_illuminations) * diff
    res = res.sum(-2)
    return res

def render_sg(lgtSG, diff, roughness, c, normal, view):    
    # Expand to dimension: [#batch_size, #num_lgtSG, #num_matSG, #dim]
    # roughness  = c[:, None, 0:1]                # [N, 1, 1]
    # metallic   = c[:, None, 1:2]                # [N, 1, 1]
    
    lgtSG  = lgtSG                              # [N, N_l, 7]
    view   = view[:, None, :]                   # [N, 1, 3]
    diff   = diff[:, None, :]                   # [N, 1, 3]
    normal = normal[:, None, :]                 # [N, 1, 3]

    # Diffuse
    diff_ret = evaluate_diffuse(sg_illuminations=lgtSG,
                                normal=normal,
                                diffuse=diff)   # [N, 3]

    c = c.view(roughness.shape[0], 1, -1, 1)       # [N, 1, 12, 3]
    normal = normal[..., None, :].repeat(1, 1, c.shape[2], 1)
    view   = view[..., None, :]
    lgtSG  = lgtSG[..., None, :]
    diff  = diff[..., None, :]
    roughness = roughness[:, None, :, None]             # [N, 1, 12]

    # Specular
    spec_ret = evaluate_specular(v=view, 
                                 nml=normal,
                                 lgtSG=lgtSG,
                                 roughness=roughness,
                                 weights  = c,
                                 base_color=diff)   # [N, 3]
    render_dict = {"spec": spec_ret,
                   "diff": diff_ret,
                   "rgb":  spec_ret + diff_ret}
    return render_dict


def render_mirror(lgtSG, diff, normal, view):    
    lgtSG  = lgtSG                              # [N, N_l, 7]
    view   = view[:, None, :]                   # [N, 1, 3]
    diff   = diff[:, None, :]                   # [N, 1, 3]
    normal = normal[:, None, :]                 # [N, 1, 3]

    lgt_dirs = reflect(-view, normal)
    lgtSG_lobe = F.normalize(lgtSG[..., :3], p=2, dim=-1)
    lgtSG_lbd  = torch.abs(lgtSG[..., 3:4])
    lgtSG_mu   = torch.abs(lgtSG[..., -3:])
    rgb    = calculate_sg_basis(lgt_dirs, lgtSG_lobe, lgtSG_lbd, lgtSG_mu)
    rgb    = torch.sum(rgb * diff, dim=-2)
    return rgb