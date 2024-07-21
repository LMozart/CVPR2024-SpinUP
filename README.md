# Spin-UP: Spin Light-for Natural Light Uncalibrated Photometric Stereo
**[Spin-UP: Spin Light-for Natural Light Uncalibrated Photometric Stereo](https://zongrui.page/CVPR2024-SpinUP/)**
<!-- <br> -->
[Zongrui Li*](https://github.com/LMozart), Zhan Lu*, Haojie Yan, [Boxin Shi](http://ci.idm.pku.edu.cn/), [Gang Pan](https://person.zju.edu.cn/en/gpan), [Qian Zheng](https://person.zju.edu.cn/zq), [Xudong Jiang](https://personal.ntu.edu.sg/exdjiang/)

Given a set of observed images captured under arbitrary spin light, we recovers surface normal of the objects.
## Dependencies
We use Anaconda to install the dependencies given following code:
```shell
# Create a new python3.8 environment
conda env create -f code/environment.yml
conda activate spinup
```

## Test
To test our method, please download the dataset and checkpoint files from [link](https://drive.google.com/drive/folders/1SMAUBmFzbv6q5o9_FHnUnugo-kvJsc2P?usp=share_link).

To test Spin-UP on particular object, please run:
```shell
python test.py --config CONFIG_PATHs
```
To test Spin-UP on multiple objects in a particular dataset, please run:
```shell
# Synthetic Dataset.
sh configs/sync-rand-light/test.sh 
# Real-world Dataset.
sh configs/real/test.sh 
```

## Train
To train on particular dataset, please run:
```shell
# Synthetic Dataset.
sh configs/sync-rand-light/train_1.sh
sh configs/sync-rand-light/train_2.sh
# Real-world Dataset.
sh configs/real/train_1.sh
sh configs/real/train_2.sh
```
To train on your own data, please first follow the steps in code/preprocess/image_process.
```shell
sh preprocess/image_process/preprocess_crop_imgs.sh
# create mask.jpg in each ${out_root}/{scene} folder with phtotoshop
sh preprocess/image_process/preprocess_offset.sh
```
Then, use code/preprocess/camera_calib/even.py to calibrate the dataset. After that, use script: code/preprocess/light_init/init_env_map_gray_fixlobe_sh.py to preprocess the light map.