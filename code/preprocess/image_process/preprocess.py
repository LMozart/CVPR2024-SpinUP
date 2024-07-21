from utils.mov2imgs import  video2imgs_intervalFrame
from utils.imgs_crop import crop_resize_imgs
import argparse
from os.path import basename,dirname,join
import glob
import os


def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--root', type=str, default='data/')
    parser.add_argument('--out_root', type=str, default= 'data_out/')
    parser.add_argument('--skip', type=int, default=2)
    parser.add_argument('--suffix', type=str, default='mov')
    args = parser.parse_args()


    mov_path_list = sorted([i for i in glob.glob(f"{args.root}/*.{args.suffix}")])
    scene_list = sorted([basename(i).replace(f'.{args.suffix}','') for i in glob.glob(f"{args.root}/*.{args.suffix}")])
    print(f"==>> scene_list: {scene_list}")

    step = 1
    # step = 4
    # step = 6

    for i in range(len(scene_list)):

        root = args.root
        mov_path = mov_path_list[i]
        # data_step2_root = join(args.root, 'data_step2', i )   # , 'obj'

        #! step1  trim mov                          out_folder : data_step1
        
        #! step2  extract_png                       out_folder : data_step2
        
        if step == 1:
            numFrames = video2imgs_intervalFrame(inPath=mov_path,outRoot=join(args.out_root,scene_list[i],'obj'),skip=args.skip)


        #! step3 create mask use ps   creat_mask    out_folder : data_step2


        #! step4 8_points_ransac                     out_folder : data
        if step == 4:

            #* 8_points_ransac
            os.makedirs( join(root,'data',f'{i}'), exist_ok=True)
            os.system(f"python 8_points_ransac.py --data_path  {data_step2_root} --focal 32 |tee {dirname(dirname(mov_path))}/data/{i}/print.log")


        #! step5 find_offset                            out_folder : data


        #! step6 crop_resize_imgs
        if step == 6:
            # outRoot2 =join(dirname(inPath), f'{dsize[0]}x{dsize[1]}', i) 
            data_root =join(root, 'data', i )   # , 'obj'
            os.makedirs(join(data_root,'obj'),exist_ok=True)
            # print('##',dirname(outRoot1))


            suffix=args.suffix
            y_lu,x_lu = 0, 420   # y left up ,  x left up
            H,W= 1080,1080   # 2160, 3840
            area = (x_lu, y_lu, x_lu+W, y_lu+H)  #x1,y1,x2,y2
            dsize=(540,540)



            #* imgs
            # outRoot3 = join(dirname(outRoot2), 'obj_shifted_masked')

            imagesPathLst = sorted([i for i in glob.glob(f"{data_step2_root}/obj_shifted_masked/*.{suffix}")])
            crop_resize_imgs(imagesPathLst=imagesPathLst, outRoot=join(data_root,'obj'), suffix=suffix, area=area , dsize=dsize)

            #* mask
            imagesPathLst = sorted([i for i in glob.glob(f"{data_step2_root}/mask.png")])
            crop_resize_imgs(imagesPathLst=imagesPathLst, outRoot=data_root, suffix=suffix, area=area , dsize=dsize)



if __name__=='__main__':
    main()