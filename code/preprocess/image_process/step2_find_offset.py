import  cv2
import numpy as np
import tqdm 
import os
import glob 
from os.path import join , dirname, basename
from utils.imgs_crop import crop_resize_imgs

from skimage.metrics import structural_similarity as ssim
import argparse





def shutil_rmtree(folder_path):
    import shutil
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"'{folder_path}' deleted")
    else:
        print(f"Folder: '{folder_path}' do not exist")


def mse_func(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err


def  find_offset(img1,img2,mask):

    mask[mask > 0] = 1
    img1_masked = img1*mask[:,:,None]

    height,width = img1.shape[:2]

    mse_min = 1e9
    ssim_max = -1   # -1~1
    offsetx = 0
    offsety = 0 

    for x in range(-20,20,1):    # -20 0
        for y in range(-10, 10):
            M = np.float32([[1,0,x],[0,1,y]])
            img2_shifted = cv2.warpAffine(img2,M,(width,height))
            img2_shifted_masked = img2_shifted * mask[:,:,None]

            mse = mse_func(img1_masked, img2_shifted_masked)

            if mse < mse_min:
                offsetx = x
                offsety = y 
                mse_min = mse

    return offsetx,offsety





def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--root', type=str, default= 'input')
    parser.add_argument('--out_root', type=str, default= 'output')
    parser.add_argument('--skip', type=int, default=2)
    parser.add_argument('--suffix', type=str, default='mov')
    args = parser.parse_args()

    scene_list = sorted([basename(i).replace(f'.{args.suffix}','') for i in glob.glob(f"{args.root}/*.{args.suffix}")])
    print(f"==>> scene_list: {scene_list}")

    for i in range(len(scene_list)):
        
        scene = scene_list[i]

        scene_root = f'{args.out_root}/{scene}'
    
        img_path_list = sorted(glob.glob(f'{scene_root}/obj/*.png'))
        N_imgs = len(img_path_list)


        img1_base = cv2.imread(f'{scene_root}/obj/000.png',-1)

        mask = cv2.imread(f'{scene_root}/mask.png',cv2.IMREAD_GRAYSCALE)
        print(f"==>> f'{scene_root}/mask.png': {f'{scene_root}/mask.png'}")
        print(f"==>> mask.max(): {mask.max()}")
        print(f"==>> mask.shape: {mask.shape}")
        mask = mask>250    # convert to bool

        height,width = mask.shape[:2]

        offset_np = np.zeros((N_imgs,2))   # 2: offsetx, offsety
            
        for j in tqdm.tqdm(range(N_imgs)):

            img2_path = img_path_list[j]
            img2 = cv2.imread(img2_path, -1)
            
            #! find_offset

            # #* 1  compute offset
            offsetx,offsety = find_offset(img1_base,img2,mask)
            print(f"==>> offsetx: {offsetx}")
            print(f"==>> offsety: {offsety}")
            offset_np[j,:] = np.array((offsetx,offsety))
            np.savetxt(f'{scene_root}/offset_np.txt',offset_np,fmt="%d %d")                 # overwrite save offset
            M = np.float32([[1,0,offsetx],[0,1,offsety]])
            img2_shifted = cv2.warpAffine(img2,M,(width,height))

            # #! shift and mask image
            save_path = f'{scene_root}/obj_shifted_masked/{j:0>3d}.png'
            save_img = img2_shifted * mask[:,:,None]
            os.makedirs(dirname(save_path),exist_ok=True)
            cv2.imwrite(save_path,save_img)

            #* crop_resize_imgs
            data_root= f'{scene_root}/obj_shifted_masked_crp'
            os.makedirs(data_root,exist_ok=True)
            imagesPathLst = sorted(glob.glob(f'{scene_root}/obj_shifted_masked/{j:0>3d}.png'))
            y_lu,x_lu = 0, 420   # y left up ,  x left up
            height1,width1= 1080,1080   # 2160, 3840
            area = (x_lu, y_lu, x_lu+width1, y_lu+height1)  #x1,y1,x2,y2
            dsize=(1080,1080)
            crop_resize_imgs(imagesPathLst=imagesPathLst, outRoot=data_root, suffix='png', area=area , dsize=dsize, mask=mask)
            
if __name__=='__main__':
    main()