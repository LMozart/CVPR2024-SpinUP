import torch
import numpy as np
import torch.nn.functional as F

# Math utils
EPS = 1e-7
def saturate(x, low=0.0, high=1.0):
    return x.clip(low, high)

def safe_exp(x):
    """The same as torch.exp(x), but clamps the input to prevent NaNs."""
    # return torch.exp(torch.minimum(x, torch.ones_like(x) * 87.5))
    return torch.exp(x)

def magnitude(x: torch.Tensor) -> torch.Tensor:
    return safe_sqrt(dot(x, x))

def dot(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    return torch.sum(x * y, dim=-1, keepdims=True)

def safe_reciprocal(x: torch.Tensor) -> torch.Tensor:
    return torch.reciprocal(torch.maximum(x, torch.ones_like(x) * EPS))

def safe_div(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.divide(x1, torch.maximum(x2, torch.ones_like(x2) * EPS))

def safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    sqrt_in = torch.maximum(x, torch.ones_like(x) * EPS)
    return torch.sqrt(sqrt_in)

def reflect(d: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return d - 2 * dot(d, n) * n

def mix(x, y, a):
    a = a.clip(0, 1)
    return x * (1 - a) + y * a

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

def axis_angle_rotation(angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    
    R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))

def axis_angle_rotation_X(angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    
    R_flat = (one, zero, zero , zero, cos, -sin, zero, sin, cos)
    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def positional_encoding(
    tensor, num_encoding_functions=6, include_input=True, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.

    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).

    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # the input tensor is added to the positional encoding if include_input=True
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)


def get_embedding_function(
    num_encoding_functions=6, include_input=True, log_sampling=True
):
    r"""Returns a lambda function that internally calls positional_encoding.
    """
    return lambda x: positional_encoding(
        x, num_encoding_functions, include_input, log_sampling
    )

def rotation_between_vectors(vec1, vec2):
    ''' Retruns rotation matrix between two vectors (for Tensor object) '''
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    # vec1.shape = [N, 3]
    # vec2.shape = [N, 3]
    batch_size = vec1.shape[0]
    
    v = torch.cross(vec1, vec2)                                                     # [N, 3, 3]

    cos = torch.bmm(vec1.view(batch_size, 1, 3), vec2.view(batch_size, 3, 1))
    cos = cos.reshape(batch_size, 1, 1).repeat(1, 3, 3)                             # [N, 3, 3]
    
    skew_sym_mat = torch.zeros(batch_size, 3, 3).to(vec1.device)
    skew_sym_mat[:, 0, 1] = -v[:, 2]
    skew_sym_mat[:, 0, 2] = v[:, 1]
    skew_sym_mat[:, 1, 0] = v[:, 2]
    skew_sym_mat[:, 1, 2] = -v[:, 0]
    skew_sym_mat[:, 2, 0] = -v[:, 1]
    skew_sym_mat[:, 2, 1] = v[:, 0]

    identity_mat = torch.zeros(batch_size, 3, 3).to(vec1.device)
    identity_mat[:, 0, 0] = 1
    identity_mat[:, 1, 1] = 1
    identity_mat[:, 2, 2] = 1

    R = identity_mat + skew_sym_mat
    R = R + torch.bmm(skew_sym_mat, skew_sym_mat) / (1 + cos).clamp(min=1e-7)
    zero_cos_loc = (cos == -1).float()
    R_inverse = torch.zeros(batch_size, 3, 3).to(vec1.device)
    R_inverse[:, 0, 0] = -1
    R_inverse[:, 1, 1] = -1
    R_inverse[:, 2, 2] = -1
    R_out = R * (1 - zero_cos_loc) + R_inverse * zero_cos_loc                       # [N, 3, 3]
    return R_out        


def _dot(dirs, lobes):
    """Calculate dot product.
    """
    return torch.sum(dirs * lobes, dim=-1, keepdim=True)


def calculate_sg_basis(view_dirs, SG_lobes, SG_lambdas, SG_mus):
    """Calculate SG basis. 
    Referenced to "All-frequency rendering of dynamic, spatially-varying reflectance".
    """
    return SG_mus * torch.exp(SG_lambdas * (_dot(view_dirs, SG_lobes) - 1.))


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
    
    phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), torch.linspace(-np.pi, np.pi, W)], indexing='ij')
    view_dirs = torch.stack([torch.sin(theta) * torch.sin(phi),
                             torch.cos(phi),
                             -torch.cos(theta) * torch.sin(phi)], dim=-1)             # [H, W, 3]
    view_dirs = view_dirs.to(lgtSGs.device)                                           # [H, W, 3]
    view_dirs = view_dirs.unsqueeze(-2)                                               # [H, W, 1, 3]
    lgtSGs = lgtSGs.view(1, 1, N_l, 5).repeat(H, W, 1, 1)                             # [H, W, N_l, 3]

    lgtSG_lobes  = F.normalize(lgtSGs[..., :3], p=2, dim=-1)                          # [H, W, N_l, 3]
    lgtSG_lambdas= torch.abs(lgtSGs[..., 3:4])                                        # [H, W, N_l, 3]
    lgtSG_mus    = torch.abs(lgtSGs[..., 4:])                                        # [H, W, N_l, 3]
    rgb    = calculate_sg_basis(view_dirs, lgtSG_lobes, lgtSG_lambdas, lgtSG_mus)     # [H, W, N_l, 3]
    rgb    = torch.sum(rgb, dim=-2)                                                   # [H, W, 3]
    envmap = rgb.reshape((H, W, 1)).clip(min=0.)                                      # [H, W, 3]
    return envmap

def totalVariation_L2(image, mask, num_rays):
    pixel_dif1 = torch.square(image[1:, :, :] - image[:-1, :, :]).sum(dim=-1) * mask[1:, :] * mask[:-1, :]
    pixel_dif2 = torch.square(image[:, 1:, :] - image[:, :-1, :]).sum(dim=-1) * mask[:, 1:] * mask[:, :-1]
    tot_var = (pixel_dif1.sum() + pixel_dif2.sum()) / num_rays
    return tot_var