import cv2 as cv
import numpy as np
import torch
import argparse
import os
from load_natural import load_natural


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str,
        default="PATH"
    )
    parser.add_argument(
        "--H", type=int,
        default=512
    )
    parser.add_argument(
        "--W", type=int,
        default=512
    )
    parser.add_argument(
        "--focal", type=float,
        default=31
    )
    parser.add_argument(
        "--fs", type=float,
        default=14.9
    )
    parser.add_argument(
        "--target_scale", type=int,
        default=2
    )
    parser.add_argument(
        "--inverse", type=int,
        default=0
    )
    cfg = parser.parse_args()
    data_dict = load_natural(cfg.data_path)

    mask = 255-data_dict["mask"]
    imgs = data_dict["images"]
    num_imgs = imgs.shape[0]
    
    num_images = imgs.shape[0]
    rel_eulers = np.linspace(0, (num_images-1) / num_images * 2 * np.pi, num_imgs)
    
    eulers = np.array(rel_eulers)
    if cfg.inverse:
        eulers = -eulers

    K = np.array([[cfg.H * cfg.focal / cfg.fs,  0.,               cfg.H // 2.],
                  [0.,               cfg.W * cfg.focal / cfg.fs,  cfg.W // 2.],
                  [0.,               0.,                      1.]])
    data_dict = {"K": K, "f": cfg.focal, "R": eulers, "fs": cfg.fs}
    np.save(os.path.join(cfg.data_path, "camera_params.npy"), data_dict)