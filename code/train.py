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
import imageio
import kornia
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

def train(input_data, idxp_dict, testing, epoch):
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
    
    valid_xy = input_data["valid_xy"][0].to(device)
    valid_xy_code = input_data["valid_xy_code"][0].to(device)

    # Material
    valid_mat_0 = mat_mlp(valid_xy_code)
    output_diff_0       = valid_mat_0["diff"]
    output_spec_coeff_0 = valid_mat_0["spec_coeff"]

    # Depths -> Normal
    valid_d_t  = nml_mlp(valid_xy_code)[..., 0]

    if not testing:
        mm_p = training_data_loader.mm_p
        valid_d_t[1] = valid_d_t[0] * (mm_p - 1) / mm_p + valid_d_t[1] / mm_p
        valid_d_t[2] = valid_d_t[0] * (mm_p - 1) / mm_p + valid_d_t[2] / mm_p
        valid_d_t[3] = valid_d_t[0] * (mm_p - 1) / mm_p + valid_d_t[3] / mm_p
        valid_d_t[4] = valid_d_t[0] * (mm_p - 1) / mm_p + valid_d_t[4] / mm_p
    
    xx_t, yy_t = valid_xy[..., 0], -valid_xy[..., 1] # [h, w]
    xxx_t = xx_t * fs / focal * valid_d_t.squeeze()
    yyy_t = yy_t * fs / focal * valid_d_t.squeeze()
    pos_t = torch.stack([xxx_t, yyy_t, valid_d_t], dim=-1) # [h, w, 3]

    top   = (pos_t[1] - pos_t[0]) / training_data_loader.mm_p
    left  = (pos_t[2] - pos_t[0]) / training_data_loader.mm_p
    down  = (pos_t[3] - pos_t[0]) / training_data_loader.mm_p
    right = (pos_t[4] - pos_t[0]) / training_data_loader.mm_p

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
    rough = sg_bases(device, epoch) # [1, num_sgs]
    rough = rough.repeat(view_dir.shape[0], 1)

    output_spec_coeff = output_spec_coeff.view(num_rays, num_bases, 1)
    output_spec_coeff = dynamic_basis(output_spec_coeff, epoch, end_epoch, num_bases)

    lgtSG_ret = light_model.get_light_from_idx(idx=input_data['item_idx'].to(device), num_rays=num_rays)
    render_dict = render_sg(lgtSG=lgtSG_ret["lgtSG"],
                            diff=output_diff, 
                            roughness=rough,
                            c=output_spec_coeff,
                            normal=output_nml, 
                            view=view_dir)
    render_rgb = render_dict["rgb"]
    if cfg.clip_render:
        render_rgb = torch.clamp(render_rgb, 0, 1)
    
    if not testing:
        img_mean  = input_data["img_mean"][0].permute(2, 0, 1).to(device)
        iter_step = (epoch - 1) * iters_per_epoch + iter_num
        rgb_loss = F.l1_loss(render_rgb * shadow_mask.view(-1, 1), 
                             gt_rgb * shadow_mask.view(-1, 1))
        rgb_loss_val = rgb_loss.item()
        loss = rgb_loss
        
        depth_map       = torch.zeros((h, w), dtype=torch.float32, device=device)
        depth_map[idxp] = valid_d_t[0]

        if epoch <= int(cfg.loss.regularize_epoches * end_epoch):  # if epoch is small, use tv to guide the network
            if cfg.loss.diff_tv_factor > 0:
                diff_color_map = torch.zeros((h, w, 1 if cfg.dataset.gray_scale else 3), dtype=torch.float32, device=device)
                diff_color_map[idxp] = output_diff_0[0, ...]
                tv_loss = totalVariation(diff_color_map, mask, num_rays) * batch_size * cfg.loss.diff_tv_factor
                loss += tv_loss
                wandb.log({"loss/diff_tv_loss": tv_loss}, step=iter_step)
            if cfg.loss.spec_tv_factor > 0:
                spec_color_map = torch.zeros((h, w, output_spec_coeff_0.size(2)), dtype=torch.float32, device=device)
                spec_color_map[idxp] = output_spec_coeff_0[0, ...]
                tv_loss = totalVariation(spec_color_map, mask, num_rays) * batch_size * cfg.loss.spec_tv_factor
                loss += tv_loss
                wandb.log({"loss/spec_tv_loss": tv_loss}, step=iter_step)
            if cfg.loss.normal_tv_factor > 0:
                nml_map = torch.zeros((h, w, 3), dtype=torch.float32, device=device)
                nml_map[idxp] = output_nml_0
                tv_loss = totalVariation_L2(nml_map, mask, num_rays) * batch_size * cfg.loss.normal_tv_factor
                loss += tv_loss
                wandb.log({"loss/normal_tv_loss": tv_loss}, step=iter_step)
        if cfg.loss.contour_factor > 0 and epoch <= int(0.75 * end_epoch):
            nml_map = torch.zeros((h, w, 3), dtype=torch.float32, device=device)
            nml_map[idxp] = output_nml_0
            tv_loss = totalVariation_L2(nml_map, mask, num_rays) * batch_size * 0.01 * cfg.loss.normal_tv_l_factor
            loss += tv_loss
            wandb.log({"loss/normal_tv_loss": tv_loss}, step=iter_step)
            wandb.log({"loss/normal_contour_loss": tv_loss}, step=iter_step)

        contour_nml_loss = 1 - torch.sum(output_nml_0 * pre_contour_nml, dim=-1).mean()
        loss += contour_nml_loss * cfg.loss.contour_factor
        euler_minus = light_model.euler[1:] - light_model.euler[:-1]
        loss += torch.nn.functional.relu(-(2 * torch.tensor(np.pi, device=device) - \
                                               torch.abs(torch.sum(euler_minus))))
        
        if cfg.loss.chrom_factor > 0:
            chrom_gt_rgb = F.normalize(gt_rgb, p=2, dim=-1) # [N, 3]
            chrom_basecolor = F.normalize(output_diff, p=2, dim=-1) # [N, 3]
            chrom_loss = ((chrom_gt_rgb - chrom_basecolor) ** 2).mean()
            loss += chrom_loss * cfg.loss.chrom_factor

            wandb.log({"loss/chrom_loss": chrom_loss}, step=iter_step)
        
        albedo_loss = torch.relu(output_diff_0 - torch.ones_like(output_diff_0)).sum()
        weight_loss = torch.relu(output_spec_coeff_0 - torch.ones_like(output_spec_coeff_0)).sum()

        loss += albedo_loss
        loss += weight_loss
        if epoch == 100:
            for g in lgt_mu_optimizer.param_groups:
                g['lr'] = 1e-3
        if epoch == 200:
            light_model.set_trainable_true()
            for g in optimizer.param_groups:
                g['lr'] = 1e-3
        if epoch == 500:
            for g in lgt_lbd_optimizer.param_groups:
                g['lr'] = 1e-3
        rots_euler = light_model.euler
        pred_rots_euler = torch.cat([torch.zeros([1, 1], device=rots_euler.device), rots_euler[1:, :]], 0)
        pred_rots_vec = torch.cat((torch.cos(pred_rots_euler), torch.sin(pred_rots_euler)), dim=-1)
        gt_rots_euler = -torch.tensor(training_data_loader.gt_rot, device=device)
        gt_rots_vec = torch.cat((torch.cos(gt_rots_euler[:, None]), torch.sin(gt_rots_euler[:, None])), dim=-1)
        avg_rot_angle = torch.mean(torch.sum(pred_rots_vec * gt_rots_vec, dim=-1))
        wandb.log({"loss/val_rot_angle": np.arccos(avg_rot_angle.detach().cpu().numpy()) / np.pi * 180}, step=iter_step)

        lgt_mu_optimizer.zero_grad()
        lgt_lbd_optimizer.zero_grad()
        mat_optimizer.zero_grad()
        optimizer.zero_grad()
        sg_bases.zero_grad()
        loss.backward()
        optimizer.step()
        sg_bases.step()
        mat_optimizer.step()
        lgt_mu_optimizer.step()
        lgt_lbd_optimizer.step()

        # log the running loss
        nml_loss = torch.arccos(torch.clamp((output_nml_0.detach() * gt_nml.detach()).sum(dim=-1), max=1, min=-1)).mean()
        
        wandb.log({"loss/train_rgb_loss": rgb_loss_val, "loss/val_nml_loss": nml_loss.item() / math.pi * 180,}, step=iter_step)

        metric_dict = {"rgb_loss": rgb_loss_val * 255, 
                       "nml_loss": nml_loss.item() / np.pi * 180}
        return metric_dict
    else:
        bounding_box_int = idxp_dict["bbox_int"]
        idxp_invalid     = np.where(mask.cpu().numpy() < 0.5)
        nml_loss = torch.arccos(torch.clamp((output_nml_0.detach() * gt_nml.detach()).sum(dim=-1), max=1, min=-1)).mean()
        print("Test MAE:", torch.arccos(nml_loss) / math.pi * 180)
        
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
        nml_err_map = torch.zeros((h, w), dtype=torch.float32, device=device)
        nml_err = torch.arccos(
            torch.clamp((output_nml_0.detach() * gt_nml.detach()).sum(dim=-1), max=1, min=-1)) / math.pi * 180
        nml_err_map[idxp] = torch.clamp(nml_err, max=90)
        nml_err_map = nml_err_map.cpu().numpy()
        nml_err_map = (np.clip(nml_err_map / 90, 0, 1) * 255).astype(np.uint8)
        nml_err_map = cv.applyColorMap(nml_err_map, colormap=cv.COLORMAP_JET)
        nml_err_map[idxp_invalid] = 255
        nml_err_map = nml_err_map[bounding_box_int[0]:bounding_box_int[1], bounding_box_int[2]:bounding_box_int[3]]
        nml_stack = np.concatenate((gt_nml_map, est_nml_map, nml_err_map), axis=1)
        cv.imwrite(os.path.join(log_path, f"nml_comp_{epoch}.png"), nml_stack)
        cv.imwrite(os.path.join(log_path, f"nml.png"), nml_err_map)

        est_nml_map = torch.zeros((h, w, 3), device=temp_est_nml.device)
        est_nml_map[idxp] = (temp_est_nml + 1) / 2
        est_nml_map = est_nml_map.cpu().numpy()
        est_nml_map = np.clip(est_nml_map * 255., 0., 255.).astype(np.uint8)[:, :, ::-1]
        est_nml_map[idxp_invalid] = 0

        cv.imwrite(os.path.join(log_path, f"nml_err_{torch.arccos(nml_loss) / math.pi * 180}.png"), est_nml_map)
        
        ######## Render Plots ########
        est_rgb = render_rgb * shadow_mask.view(-1, 1)
        est_rgb = process_images(pixels=est_rgb[:len(idxp[0])],
                                 h=h, w=w, channel=3,
                                 idxp=idxp,
                                 idxp_invalid=idxp_invalid,
                                 bounding_box_int=bounding_box_int)

        gt_rgb  = process_images(pixels=gt_rgb[:len(idxp[0])],
                                 h=h, w=w, channel=3,
                                 idxp=idxp,
                                 idxp_invalid=idxp_invalid,
                                 bounding_box_int=bounding_box_int)
        
        est_rgb_no_shd = process_images(pixels=render_rgb[:len(idxp[0])],
                                        h=h, w=w, channel=3,
                                        idxp=idxp,
                                        idxp_invalid=idxp_invalid,
                                        bounding_box_int=bounding_box_int)
        
        est_shadow = process_images(pixels=shadow_mask.view(-1, 1)[:len(idxp[0])],
                                h=h, w=w, channel=3,
                                idxp=idxp,
                                idxp_invalid=idxp_invalid,
                                bounding_box_int=bounding_box_int)

        diff_map  = process_images(pixels=output_diff_0[0] / output_diff_0[0].max(),
                                   h=h, w=w, channel=3,
                                   idxp=idxp,
                                   idxp_invalid=idxp_invalid,
                                   bounding_box_int=bounding_box_int)
        
        meta = output_spec_coeff_0[0, ..., :1] / output_spec_coeff_0[0, ..., :1].max()
        meta = meta.repeat(1, 1, 3)
        meta_map  = process_images(pixels=meta,
                                   h=h, w=w, channel=3,
                                   idxp=idxp,
                                   idxp_invalid=idxp_invalid,
                                   bounding_box_int=bounding_box_int)
        
        roug = output_spec_coeff_0[0, ..., 1:2]  / output_spec_coeff_0[0, ..., 1:2].max()
        roug = roug.repeat(1, 1, 3)
        roug_map  = process_images(pixels=roug,
                                   h=h, w=w, channel=3,
                                   idxp=idxp,
                                   idxp_invalid=idxp_invalid,
                                   bounding_box_int=bounding_box_int)
        
        render_stack = np.concatenate((gt_rgb, est_rgb, est_rgb_no_shd, diff_map, est_shadow, meta_map, roug_map), axis=1)
        cv.imwrite(os.path.join(log_path, f"render_comp_{epoch}.png"), render_stack)
        
        ######## Light Plots ########
        est_lgtSG_ret = light_model.get_light_from_idx(torch.tensor([10], device=device), num_rays)
        est_env_map = render_envmap(lgtSGs=est_lgtSG_ret["lgtSG"].detach().cpu()[0], H=256, W=512)
        est_env_map = est_env_map.numpy().astype(np.float32) / 10
        
        gt_env_map  = input_data['lgt'][0, ...]
        gt_env_map  = gt_env_map.numpy().astype(np.float32)

        light_stack = np.concatenate([cv.resize(gt_env_map, (512, 256)), est_env_map], axis=1)
        write_exr(os.path.join(log_path, f'light_comp_{epoch}.exr'), light_stack)
        nml_stack    = wandb.Image(nml_stack[..., ::-1], caption="GT_est_err")
        render_stack = wandb.Image(render_stack[..., ::-1], caption="GT_render_shading_bcol_shad_matal_rh")
        light_stack  = wandb.Image(np.clip(light_stack, 0, 1), caption="GT_est")
        
        wandb.log({"normal comparsion": nml_stack})
        wandb.log({"render comparsion": render_stack})
        wandb.log({"lighting comparsion": light_stack})

        # write_exr(os.path.join(log_path, f'light_est.exr'), est_env_map)


if __name__ == "__main__":
    ######################
    ''' Set up configs.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sync-lzr-abl1/shape_buddha.yml", help="Path to (.yml) config file.")
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
    time_str = time.strftime('%Y%m%d-%H%M%S',time.localtime(time.time()))
    log_path += f"@{time_str}"
    data_path = os.path.expanduser(cfg.dataset.data_path)
    lgt_path = os.path.expanduser(cfg.dataset.lgt_path)

    os.makedirs(os.path.join(log_path, "code_backup"), exist_ok=True)

    cfg.loss.chrom_factor = 0.5
    cfg.loss.shadow_factor = 0.01
    cfg.use_shadow = False
    cfg.use_pre_shadow = True
    cfg.clip_render = False
    print(f'Chrom Loss: {cfg.loss.chrom_factor}, Shadow Loss: {cfg.loss.shadow_factor}, Use Shadow: {cfg.use_shadow}, Clip Render: {cfg.clip_render}')
    cfg.num_sg = 64
    cfg.shadow_gaussian_filter_size = 5
    cfg.image_downscale = 1

    if configargs.testing:
        writer = None
    else:
        # writer = SummaryWriter(log_path)  # tensorboard --logdir=runs
        run = wandb.init(project='spinup', name=os.path.basename(configargs.config)[:-4], config=cfg, tags=["spinup"])
        wandb.run.log_code(".")
        copyfile(__file__, os.path.join(log_path, 'code_backup/train.py'))
        copyfile("./code/models.py", os.path.join(log_path, 'code_backup/models.py'))
        copyfile("./code/render_sg.py", os.path.join(log_path, 'code_backup/render_sg.py'))
        copyfile("./code/dataloader.py", os.path.join(log_path, 'code_backup/dataloader.py'))
        copyfile("./code/load_natural.py", os.path.join(log_path, 'code_backup/load_natural.py'))
        copyfile(configargs.config, os.path.join(log_path, 'code_backup/config.yml'))

    start_epoch = cfg.experiment.start_epoch
    end_epoch = cfg.experiment.end_epoch
    batch_size = int(eval(cfg.experiment.batch_size))
    
    ##########################
    ''' Build train data loader.
    '''
    if 'Real' in cfg.dataset.data_path or 'real' in cfg.dataset.data_path:
        input_data_dict = load_real(data_path, lgt_path, cfg, scale=cfg.image_downscale)
    else:
        input_data_dict = load_natural(data_path, lgt_path, cfg, scale=cfg.image_downscale)

    training_data_loader = Data_Loader(
        input_data_dict,
        gray_scale=cfg.dataset.gray_scale,
        data_len=300,
        mode='testing',
        shadow_threshold=cfg.dataset.shadow_threshold,
        log_path=log_path
    )
    training_dataloader = torch.utils.data.DataLoader(training_data_loader, batch_size=batch_size, shuffle=not configargs.testing, num_workers=0)

    # 0. Parameters.
    focal = training_data_loader.focal
    fs    = training_data_loader.fs / 2.
    euler = training_data_loader.euler
    
    # 1. mask and related.
    num_imgs = training_data_loader.num_images

    # 2. bounding box for plots.

    # Build test data loader.
    eval_data_len = len(training_data_loader) if configargs.testing else 1
    if configargs.quick_testing:
        eval_data_len = 1
        configargs.testing = True
    if cfg.experiment.eval_every_iter <= (end_epoch-start_epoch+1):
        eval_data_loader = Data_Loader(
            input_data_dict,
            gray_scale=cfg.dataset.gray_scale,
            data_len=eval_data_len,
            mode='valid',
            shadow_threshold=cfg.dataset.shadow_threshold,
            log_path=log_path
        )
        eval_dataloader = torch.utils.data.DataLoader(eval_data_loader, batch_size=1, shuffle=False, num_workers=0)
    
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
        SG_init=training_data_loader.lgt_SG,
        euler_init=euler
    )   
    light_model.train()
    light_model.to(device)


    params_list = []
    params_list.append({'params': nml_mlp.parameters()})
    
    optimizer = optim.Adam(params_list, lr=cfg.optimizer.lr)
    mat_optimizer = optim.Adam([{'params': mat_mlp.parameters()},
                                {'params': light_model.lgtSG_mu}], 
                                lr=cfg.optimizer.lr)
    lgt_mu_optimizer = optim.Adam([{'params': light_model.lgtSG_lobe}], 
                                           lr=1e-3)
    lgt_lbd_optimizer = optim.Adam([{'params': light_model.lgtSG_lbd}], 
                                           lr=1e-3)

    scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                          step_size = 1000, 
                                          gamma     = 0.1)
    mat_scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                              step_size = 1500, 
                                              gamma     = 0.1)

    lgt_mu_scheduler = optim.lr_scheduler.StepLR(lgt_mu_optimizer,
                                                step_size = 1500, 
                                                gamma     = 0.1)
    lgt_lbd_scheduler = optim.lr_scheduler.StepLR(lgt_mu_optimizer,
                                                step_size = 1500, 
                                                gamma     = 0.1)
    

    ##########################
    ''' Load checkpoints.
    '''
    if configargs.testing:
        cfg.models.load_checkpoint = True
        cfg.models.checkpoint_path = log_path
    
    # ckpt = torch.load("./code/test_512-512-0_1.pth", map_location=device)
    # nml_mlp.load_state_dict(ckpt['nml_mlp_state_dict'])

    if cfg.models.load_checkpoint:
        model_checkpoint_pth = os.path.expanduser(cfg.models.checkpoint_path)
        if model_checkpoint_pth[-4:] != '.pth':
            model_checkpoint_pth = sorted(glob.glob(os.path.join(model_checkpoint_pth, 'model*.pth')))[-1]
        print('Found checkpoints', model_checkpoint_pth)

        ckpt = torch.load(model_checkpoint_pth, map_location=device)
        nml_mlp.load_state_dict(ckpt['nml_mlp_state_dict'])
        mat_mlp.load_state_dict(ckpt['mat_mlp_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        light_model.load_state_dict(ckpt['light_model_state_dict'])
        start_epoch = ckpt['global_step'] + 1
    
    if configargs.testing:
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

    ''' Start Training / Testing.
    '''
    start_t = time.time()
    iters_per_epoch = len(training_dataloader)

    epoch = 0
    nml_mlp.eval()
    mat_mlp.eval()

    idxp_dict = eval_data_loader.get_idxp_and_mask("valid")
    with torch.no_grad():
        for eval_idx, eval_datain in enumerate(eval_dataloader, start=0):
            batch_size = 1
            train(input_data=eval_datain, idxp_dict=idxp_dict, testing=True, epoch=epoch)
    nml_mlp.train()
    mat_mlp.train()

    idxp_dict = training_data_loader.get_idxp_and_mask("test")
    with trange(start_epoch, end_epoch+1) as pbar:
        for epoch in pbar:
            for iter_num, input_data in enumerate(training_dataloader):
                if not configargs.testing:
                    batch_size = int(eval(cfg.experiment.batch_size))
                    metric_dict = train(input_data=input_data, idxp_dict=idxp_dict, testing=False, epoch=epoch)
                    
                    rgb_loss = metric_dict["rgb_loss"]
                    nml_loss = metric_dict["nml_loss"]
                    pbar.set_description(f"loss:{rgb_loss:.8f}, MAE:{nml_loss:.2f}")
            scheduler.step()
            mat_scheduler.step()
            lgt_mu_scheduler.step()
            lgt_lbd_scheduler.step()
            sg_bases.step_scheduler()
            training_data_loader.resample_patch_idx()
            idxp_dict = training_data_loader.get_idxp_and_mask("test")
            if (epoch == 0.05 * end_epoch) or (epoch == 0.1 * end_epoch):
                training_data_loader.shrink_scale()
            # print(optimizer.param_groups[0]['lr'])

            if epoch % cfg.experiment.save_every_epoch == 0:
                savepath = os.path.join(log_path, 'model_params_%05d.pth' % epoch)
                torch.save({
                    'global_step': epoch,
                    'nml_mlp_state_dict':     nml_mlp.state_dict(),
                    'mat_mlp_state_dict':     mat_mlp.state_dict(),
                    'light_model_state_dict': light_model.state_dict(),
                    'sg_state_dict':          sg_bases.state_dict(),
                    'optimizer_state_dict':   optimizer.state_dict(),
                    'scheduler_state_dict':   scheduler.state_dict(),
                }, savepath)
                print('Saved checkpoints at', savepath)

            if epoch % cfg.experiment.eval_every_iter == 0:
                nml_mlp.eval()
                mat_mlp.eval()
                eval_idxp_dict = eval_data_loader.get_idxp_and_mask("valid")
                with torch.no_grad():
                    for eval_idx, eval_datain in enumerate(eval_dataloader, start=0):
                        batch_size = 1
                        train(input_data=eval_datain, idxp_dict=eval_idxp_dict, testing=True, epoch=epoch)
                nml_mlp.train()
                mat_mlp.train()