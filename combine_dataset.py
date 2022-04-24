import os
import cv2
import numpy as np
import shutil


def combine_data(syn_rgb_dir, syn_seg_dir, front_rgb_dir, front_seg_dir, save_rgb, save_seg):

    syn_rgb_files = os.listdir(syn_rgb_dir)
    syn_seg_files = os.listdir(syn_seg_dir)
    
    front_rgb_files = os.listdir(front_rgb_dir)
    front_seg_files = os.listdir(front_seg_dir)
    
    for i, (rgb, seg) in enumerate(zip(syn_rgb_files, syn_seg_files)):
        src = os.path.join(syn_rgb_dir, rgb)
        dst = os.path.join(save_rgb, "syn_"+rgb)
        shutil.copyfile(src, dst)
        
        src = os.path.join(syn_seg_dir, rgb)
        dst = os.path.join(save_seg, "syn_"+seg)
        shutil.copyfile(src, dst)

    for i, (rgb, seg) in enumerate(zip(front_rgb_files, front_seg_files)):
        src = os.path.join(front_rgb_dir, rgb)
        dst = os.path.join(save_rgb, "front_"+rgb)
        shutil.copyfile(src, dst)
        
        src = os.path.join(front_seg_dir, rgb)
        dst = os.path.join(save_seg, "front_"+seg)
        shutil.copyfile(src, dst)
    

if __name__ == "__main__":
    syn_rgb_dir = '/home/adithyas/atdt/data/carla/front/rgb'
    syn_seg_dir = '/home/adithyas/atdt/data/carla/front/seg'
   
    front_rgb_dir = '/home/adithyas/atdt/data/nuImages/mini/front/rgb'
    front_seg_dir = '/home/adithyas/atdt/data/nuImages/mini/front/seg'
 
    save_rgb = '/home/adithyas/atdt/data/dataset/rgb'
    save_seg = '/home/adithyas/atdt/data/dataset/seg'
    
    combine_data(syn_rgb_dir, syn_seg_dir, front_rgb_dir, front_seg_dir, save_rgb, save_seg)