import os
import numpy as np
import cv2
from carla_nuscenes_map import NUSCENES_CARLA_MAP

def map_nuscenes_to_carla(front_seg_dir, save_dir):
    images = os.listdir(front_seg_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for i, image in enumerate(images):
        print(f"{i}")
        img = cv2.imread(os.path.join(front_seg_dir, image))
        img_copy = img.copy()
        for key, value in NUSCENES_CARLA_MAP.items():
            img_copy[img==key] = value
        cv2.imwrite(os.path.join(save_dir, image), img_copy)   

if __name__ == "__main__":
    front_seg_dir = '/home/adithyas/atdt/data/nuImages/mini/front/seg_viz'
    save_dir = '/home/adithyas/atdt/data/nuImages/mini/front/seg'
    map_nuscenes_to_carla(front_seg_dir, save_dir)