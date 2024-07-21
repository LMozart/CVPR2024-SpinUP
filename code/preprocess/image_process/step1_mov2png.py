from utils.mov2imgs import  video2imgs_intervalFrame
import argparse
from os.path import basename,dirname,join
import glob
import os


'''
data 

'''

def main():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--root', type=str, default= 'data/indoor')
    parser.add_argument('--out_root', type=str, default= 'data_out/indoor')
    parser.add_argument('--skip', type=int, default=2)
    parser.add_argument('--suffix', type=str, default='mov')
    args = parser.parse_args()


    mov_path_list = sorted([i for i in glob.glob(f"{args.root}/*.{args.suffix}")])
    scene_list = sorted([basename(i).replace(f'.{args.suffix}','') for i in glob.glob(f"{args.root}/*.{args.suffix}")])
    print(f"==>> scene_list: {scene_list}")

    for i in range(len(scene_list)):

        mov_path = mov_path_list[i]
        _ = video2imgs_intervalFrame(inPath=mov_path,outRoot=join(args.out_root,scene_list[i],'obj'),skip=args.skip)


if __name__=='__main__':
    main()