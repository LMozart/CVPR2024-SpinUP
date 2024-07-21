import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import glob
from tqdm import trange, tqdm
import cv2 as cv
import yaml
import math
import argparse
import random

from shutil                  import copyfile
# from torch.utils.tensorboard import SummaryWriter

from models           import *
from cfgnode          import CfgNode
from dataloader       import Data_Loader
from load_natural     import load_natural, load_real
from position_encoder import get_embedding_function

from utils      import *

from render_sg import render_sg, render_envmap

from io_exr import write_exr
import wandb

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def test(input_data, idxp_dict, testing, count):
    mask = idxp_dict["mask"].to(device)
    h, w = mask.shape
    idxp = idxp_dict["idxp"]

    batch_size = input_data['rgb'].size(0)
    num_rays   = len(idxp[0]) * batch_size

    gt_nml     = input_data['normal'][0].to(device)    # [N_pts, 3]
    if cfg.use_pre_shadow:
        shadow_mask= input_data['shadow_mask'].to(device)
    gt_rgb = input_data['rgb'].view(-1, 1 if cfg.dataset.gray_scale else 3).to(device) # [N, 3]

    view_dir = input_data["view"].view(-1, 3).to(device)

    if cfg.loss.contour_factor > 0:
        pre_contour_nml = input_data['contour_normal'][0].to(device)    # [N_pts, 3]
    
    valid_xy = input_data["valid_xy_code"][0].to(device)

    # Material
    valid_mat_0 = mat_mlp(valid_xy)
    output_diff_0       = valid_mat_0["diff"]
    output_spec_coeff_0 = valid_mat_0["spec_coeff"]

    # Depths -> Normal
    valid_d_t  = nml_mlp(valid_xy)[..., 0]

    xx_t, yy_t = valid_xy[..., 0], -valid_xy[..., 1] # [h, w]
    xxx_t = xx_t * fs / focal * valid_d_t.squeeze()
    yyy_t = yy_t * fs / focal * valid_d_t.squeeze()
    pos_t = torch.stack([xxx_t, yyy_t, valid_d_t], dim=-1) # [h, w, 3]

    top   = (pos_t[1] - pos_t[0])
    left  = (pos_t[2] - pos_t[0])
    down  = (pos_t[3] - pos_t[0])
    right = (pos_t[4] - pos_t[0])

    nml_sq1 = torch.cross(top, left, dim=-1)
    nml_sq2 = torch.cross(left, down, dim=-1)
    nml_sq3 = torch.cross(down, right, dim=-1)
    nml_sq4 = torch.cross(right, top, dim=-1)
    
    nml_sq1 = F.normalize(nml_sq1, p=2, dim=-1)
    nml_sq2 = F.normalize(nml_sq2, p=2, dim=-1)
    nml_sq3 = F.normalize(nml_sq3, p=2, dim=-1)
    nml_sq4 = F.normalize(nml_sq4, p=2, dim=-1)

    sq1_dist = 1 / ((torch.abs(top[..., -1]) + torch.abs(left[..., -1])) + 1e-6)
    sq2_dist = 1 / ((torch.abs(left[..., -1]) + torch.abs(down[..., -1])) + 1e-6)
    sq3_dist = 1 / ((torch.abs(down[..., -1]) + torch.abs(right[..., -1])) + 1e-6)
    sq4_dist = 1 / ((torch.abs(right[..., -1]) + torch.abs(top[..., -1])) + 1e-6)

    sq_sum = sq1_dist + sq2_dist + sq3_dist + sq4_dist

    sq1_ratio = (sq1_dist / sq_sum).detach()
    sq2_ratio = (sq2_dist / sq_sum).detach()
    sq3_ratio = (sq3_dist / sq_sum).detach()
    sq4_ratio = (sq4_dist / sq_sum).detach()

    est_nml = sq1_ratio[..., None] * nml_sq1 + \
              sq2_ratio[..., None] * nml_sq2 + \
              sq3_ratio[..., None] * nml_sq3 + \
              sq4_ratio[..., None] * nml_sq4

    output_nml_0 = F.normalize(est_nml, p=2, dim=-1)
    nx, ny, nz = output_nml_0[..., 0], output_nml_0[..., 1], output_nml_0[..., 2]
    output_nml_0 = torch.stack([nx, ny, -nz], dim=-1)
    
    output_diff       = output_diff_0[0].repeat(batch_size, 1)               # [N, 3]
    output_spec_coeff = output_spec_coeff_0[0].repeat(batch_size, 1)         # [N, 6]
    output_nml        = output_nml_0.repeat(batch_size, 1)                # [N, 3]

    # Rendering SG
    rough = sg_bases(device, current_epoch=2000) # [1, num_sgs]
    rough = rough.repeat(view_dir.shape[0], 1)

    output_spec_coeff = output_spec_coeff.view(num_rays, num_bases, 1)
    # if not testing:
    #     output_spec_coeff = dynamic_basis(output_spec_coeff, epoch, end_epoch, num_bases)

    lgtSG_ret = light_model.get_light_from_idx(idx=input_data['item_idx'].to(device), num_rays=num_rays)
    render_dict = render_sg(lgtSG=lgtSG_ret["lgtSG"],
                            diff=output_diff, 
                            roughness=rough,
                            c=output_spec_coeff,
                            normal=output_nml, 
                            view=view_dir)
    render_rgb = render_dict["rgb"]
    render_spe = render_dict["spec"]
    
    # Testing & Plotting
    nml_loss = torch.arccos(torch.clamp((output_nml_0.detach() * gt_nml.detach()).sum(dim=-1), max=1, min=-1)).mean()
    print(f"Test MAE:{nml_loss.mean() * 180 / math.pi}")

    bounding_box_int = idxp_dict["bbox_int"]
    idxp_invalid     = np.where(mask.cpu().numpy() < 0.5)
    
    ######## Normal Estimation Plots ########
    temp_est_nml = output_nml_0.detach().clone()
    temp_gt_nml  = gt_nml.detach().clone()
    
    est_nml_map = process_images(pixels=(temp_est_nml + 1) / 2,
                    h=h, w=w, channel=3,
                    idxp=idxp,
                    idxp_invalid=idxp_invalid,
                    bounding_box_int=bounding_box_int)
    gt_nml_map  = process_images(pixels=(temp_gt_nml + 1) / 2,
                    h=h, w=w, channel=3,
                    idxp=idxp,
                    idxp_invalid=idxp_invalid,
                    bounding_box_int=bounding_box_int)
    
    process_image  = process_images(pixels=render_rgb,
                    h=h, w=w, channel=3,
                    idxp=idxp,
                    idxp_invalid=idxp_invalid,
                    bounding_box_int=bounding_box_int)

    diff_map  = process_images(pixels=output_diff_0[0] / output_diff_0[0].max(),
                                   h=h, w=w, channel=3,
                                   idxp=idxp,
                                   idxp_invalid=idxp_invalid,
                                   bounding_box_int=bounding_box_int)
    
    spec_map  = process_images(pixels=render_spe / render_spe.max(),
                                   h=h, w=w, channel=3,
                                   idxp=idxp,
                                   idxp_invalid=idxp_invalid,
                                   bounding_box_int=bounding_box_int)

    cv.imwrite(os.path.join(log_path, f"imgs/imgs_{count}.png"), process_image)
    
    nml_err_map = torch.zeros((h, w), dtype=torch.float32, device=device)
    nml_err = torch.arccos(
        torch.clamp((output_nml_0.detach() * gt_nml.detach()).sum(dim=-1), max=1, min=-1)) / math.pi * 180
    nml_err_map[idxp] = torch.clamp(nml_err, max=90)
    nml_err_map = nml_err_map.cpu().numpy()
    nml_err_map = (np.clip(nml_err_map / 90, 0, 1) * 255).astype(np.uint8)
    nml_err_map = cv.applyColorMap(nml_err_map, colormap=cv.COLORMAP_JET)
    nml_err_map[idxp_invalid] = 255
    nml_err_map = nml_err_map[bounding_box_int[0]:bounding_box_int[1], bounding_box_int[2]:bounding_box_int[3]]
    cv.imwrite(os.path.join(log_path, f"nml_err.png"), nml_err_map)

    est_nml_map = torch.zeros((h, w, 3), device=temp_est_nml.device)
    est_nml_map[idxp] = (temp_est_nml + 1) / 2
    est_nml_map = est_nml_map.cpu().numpy()
    est_nml_map = np.clip(est_nml_map * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
    est_nml_map[idxp_invalid] = 255
    est_nml_map = est_nml_map[bounding_box_int[0]:bounding_box_int[1], bounding_box_int[2]:bounding_box_int[3]]

    cv.imwrite(os.path.join(log_path, f"nml.png"), est_nml_map)
    cv.imwrite(os.path.join(log_path, f"albedo.png"), diff_map)
    cv.imwrite(os.path.join(log_path, f"spec.png"), spec_map)
        
    ######## Light Plots ########
    est_lgtSG_ret = light_model.get_light_from_idx(torch.tensor([10], device=device), num_rays)
    est_env_map = render_envmap(lgtSGs=est_lgtSG_ret["lgtSG"].detach().cpu()[0], H=256, W=512)
    est_env_map = est_env_map.numpy().astype(np.float32) / 10
    
    gt_env_map  = input_data['lgt'][0, ...]
    gt_env_map  = gt_env_map.numpy().astype(np.float32)

    light_stack = np.concatenate([cv.resize(gt_env_map, (512, 256)), est_env_map], axis=1)
    write_exr(os.path.join(log_path, f'light_comp.exr'), est_env_map)
    

if __name__ == "__main__":
    ######################
    ''' Set up configs.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="PATH", help="Path to (.yml) config file.")
    parser.add_argument("--model_pth", type=str, default="PATH", help="Path to (.yml) config file.")
    parser.add_argument("--testing", type=str2bool, default=False, help="Enable testing mode.")
    parser.add_argument("--cuda", type=str, default='1', help="Cuda ID.")
    parser.add_argument("--quick_testing", type=str2bool, default=False, help="Enable quick_testing mode.")
    configargs = parser.parse_args()

    if configargs.quick_testing:
        configargs.testing = True
    
    ######################
    ''' Read config files.
    '''
    configargs.config = os.path.expanduser(configargs.config)
    with open(configargs.config, "r") as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    if cfg.experiment.randomseed is not None:
        random.seed(cfg.experiment.randomseed)
        torch.backends.cudnn.deterministic = True
        np.random.seed(cfg.experiment.randomseed)
        torch.manual_seed(cfg.experiment.randomseed)
        torch.cuda.manual_seed_all(cfg.experiment.randomseed)
        print(f"Using random seed: {cfg.experiment.randomseed}")
    if configargs.cuda is not None:
        cfg.experiment.cuda = "cuda:" + configargs.cuda
    device = torch.device(cfg.experiment.cuda)

    log_path = os.path.expanduser(cfg.experiment.log_path)
    data_path = os.path.expanduser(cfg.dataset.data_path)
    lgt_path = os.path.expanduser(cfg.dataset.lgt_path)

    cfg.loss.chrom_factor = 0.5
    cfg.loss.shadow_factor = 0.01
    cfg.use_shadow = False
    cfg.use_pre_shadow = True
    cfg.clip_render = False
    print(f'Chrom Loss: {cfg.loss.chrom_factor}, Shadow Loss: {cfg.loss.shadow_factor}, Use Shadow: {cfg.use_shadow}, Clip Render: {cfg.clip_render}')
    cfg.num_sg = 64
    cfg.shadow_gaussian_filter_size = 5
    cfg.image_downscale = 1

    batch_size = int(eval(cfg.experiment.batch_size))
    
    # Build test data loader.
    eval_data_len = 50
    if configargs.quick_testing:
        eval_data_len = 1
        configargs.testing = True
    
    ##########################
    ''' Build train data loader.
    '''
    if 'Real' in cfg.dataset.data_path or 'real' in cfg.dataset.data_path:
        input_data_dict = load_real(data_path, lgt_path, cfg, scale=cfg.image_downscale, test=configargs.testing)
    else:
        input_data_dict = load_natural(data_path, lgt_path, cfg, scale=cfg.image_downscale, test=configargs.testing)

    
    eval_data_loader = Data_Loader(
        input_data_dict,
        gray_scale=cfg.dataset.gray_scale,
        data_len=eval_data_len,
        mode='valid',
        shadow_threshold=cfg.dataset.shadow_threshold,
    )
    eval_dataloader = torch.utils.data.DataLoader(eval_data_loader, batch_size=1, shuffle=False, num_workers=0)
     # 0. Parameters.
    focal = eval_data_loader.focal
    fs    = eval_data_loader.fs / 2.
    euler = eval_data_loader.euler
    
    # 1. mask and related.
    num_imgs = eval_data_loader.num_images


    ##########################
    ''' Build model.
    '''
    # 4. Material Bases
    num_bases = cfg.models.specular.num_bases

    sg_bases = SG(num_bases = num_bases,
                  k_low     = cfg.models.specular.k_low,
                  k_high    = cfg.models.specular.k_high,
                  trainable_k=True)
    sg_bases.to(device)

    # 1. NeRF
    if cfg.models.use_mean_var:
        cfg.models.nerf.include_input_input += 2 if cfg.dataset.gray_scale else 6

    encode_fn_input = get_embedding_function(num_encoding_functions=cfg.models.nml_mlp.num_encoding_fn_input)
    encode_fn_input_mat = get_embedding_function(num_encoding_functions=cfg.models.mat_mlp.num_encoding_fn_input)

    nml_mlp = NormalMLP(
        num_layers =cfg.models.nml_mlp.num_layers,
        hidden_size=cfg.models.nml_mlp.hidden_size,
        skip_connect_every   =cfg.models.nml_mlp.skip_connect_every,
        num_encoding_fn_input=cfg.models.nml_mlp.num_encoding_fn_input,
        include_input_input =cfg.models.nml_mlp.include_input_input,
        encode_fn =encode_fn_input,
    )
    nml_mlp.train()
    nml_mlp.to(device)

    m_encode_fn_input = get_embedding_function(num_encoding_functions=cfg.models.mat_mlp.num_encoding_fn_input)

    mat_mlp = MaterialMLP(
        num_layers =cfg.models.mat_mlp.num_layers,
        hidden_size=cfg.models.mat_mlp.hidden_size,
        skip_connect_every   =cfg.models.mat_mlp.skip_connect_every,
        num_encoding_fn_input=cfg.models.mat_mlp.num_encoding_fn_input,
        include_input_input =cfg.models.mat_mlp.include_input_input,
        encode_fn =m_encode_fn_input,
        output_ch =num_bases
    )
    mat_mlp.train()
    mat_mlp.to(device)

    # 3. Light SG
    light_model = Light_Model_SG(
        num_sgs=cfg.num_sg,
        num_imgs=num_imgs,
        device=device,
        SG_init=eval_data_loader.lgt_SG,
        euler_init=euler
    )   
    light_model.eval()
    light_model.to(device)

    ##########################
    ''' Load checkpoints.
    '''
    if configargs.testing:
        cfg.models.load_checkpoint = True
        cfg.models.checkpoint_path = log_path
        # log_path = configargs.model_pth
        log_path = sorted(glob.glob(cfg.models.checkpoint_path + '@*'))[-1]
        dirs = os.listdir(log_path)[-1]
        os.makedirs(os.path.join(log_path, "imgs"), exist_ok=True)

    if cfg.models.load_checkpoint:
        # model_checkpoint_pth = os.path.expanduser(os.path.join(cfg.models.checkpoint_path, dirs))
        # if model_checkpoint_pth[-4:] != '.pth':
        #     model_checkpoint_pth = sorted(glob.glob(os.path.join(model_checkpoint_pth, 'model*.pth')))[-1]
        model_checkpoint_pth = log_path
        model_checkpoint_pth = sorted(glob.glob(os.path.join(model_checkpoint_pth, 'model*.pth')))[-1]
        print('Found checkpoints', model_checkpoint_pth)

        ckpt = torch.load(model_checkpoint_pth, map_location=device)
        nml_mlp.load_state_dict(ckpt['nml_mlp_state_dict'])
        mat_mlp.load_state_dict(ckpt['mat_mlp_state_dict'])
        light_model.load_state_dict(ckpt['light_model_state_dict'])
        sg_bases.load_state_dict(ckpt['sg_state_dict'])
    
    start_epoch = 1
    end_epoch = 1
    cfg.experiment.eval_every_iter = 1
    cfg.experiment.save_every_iter = 100
    
    if configargs.quick_testing:
        cfg.experiment.eval_every_iter = 100000000

    if cfg.loss.rgb_loss == 'l1':
        rgb_loss_function = F.l1_loss
    elif cfg.loss.rgb_loss == 'l2':
        rgb_loss_function = F.mse_loss
    elif cfg.loss.rgb_loss == 'sml1':
        rgb_loss_function = F.smooth_l1_loss
    else:
        raise AttributeError('Undefined rgb loss function.')

    ''' Start Testing.
    '''
    start_t = time.time()
    iters_per_epoch = len(eval_dataloader)

    epoch = 0
    nml_mlp.eval()
    mat_mlp.eval()
    count = 0

    idxp_dict = eval_data_loader.get_idxp_and_mask("valid")
    with torch.no_grad():
        for eval_idx, eval_datain in enumerate(eval_dataloader, start=0):
            batch_size = 1
            test(input_data=eval_datain, idxp_dict=idxp_dict, testing=True, count=count)
            count += 1