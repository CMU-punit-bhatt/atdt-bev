import numpy as np
import torch
from nuimages import NuImages
import matplotlib.pyplot as plt
import os
import zipfile
import cv2
import shutil
import argparse
from tqdm import tqdm
from carla_nuscenes_map import NUSCENES_CARLA_MAP


def create_dir(folder):
    if not os.path.isdir(folder):
        os.makedirs(folder)


class NuimagesDataset:
    def __init__(self, save_folder="/content/train", type="front", dataroot='data/sets/nuimages', version="v1.0-train", sensor="CAM_FRONT", resize="resize", num_images_to_save=7500):
        self.save_folder = save_folder
        self.type = type
        self.dataroot = dataroot
        self.version = version
        self.sensor = sensor
        self.num_images_to_save = num_images_to_save
        self.nuim = NuImages(
            dataroot=dataroot, version=version, verbose=True, lazy=True)
        self.num_samples = len(self.nuim.sample)
        self.rgb = os.path.join(self.save_folder, self.type, "rgb")
        self.seg_nuscenes = os.path.join(self.save_folder, self.type, "seg_nuscenes")
        self.seg_carla = os.path.join(self.save_folder, self.type, "seg")
        self.resize = resize
        create_dir(self.rgb)
        create_dir(self.seg_nuscenes)
        create_dir(self.seg_carla)
        self.front_images = []

    def fetch_front_images(self, idx_save_path="front_idx.npy"):
        '''
        Save indices of front camera images
        Args:
        idx_save_path (str): path to store indices of front camera images (optional - only for backup)
        '''
        for sample_idx in tqdm(range(self.num_samples)):
            samples = self.nuim.sample[sample_idx]
            key_camera_token = samples['key_camera_token']
            sample_data = self.nuim.get('sample_data', key_camera_token)

            if self.sensor not in sample_data['filename']:
               continue

        self.front_images.append(sample_idx)

        self.front_images = np.array(self.front_images)
        np.save(idx_save_path, self.front_images)

    def crate_dataset(self):
        '''
        Create Nuscenes segmentation dataset
        '''
        # fetch front image indices
        if len(self.front_images) == 0:
            self.fetch_front_images()

        # define width of crop
        w = 512 
        
        # save only front camera images
        for sample_idx in tqdm(self.front_images):
            # fetch front images from Nuscenes dataset
            samples = self.nuim.sample[sample_idx]
            key_camera_token = samples['key_camera_token'] 
            sample_data = self.nuim.get('sample_data', key_camera_token)
            img_path = os.path.join(self.dataroot, sample_data['filename'])
            filename = f"{sample_idx:06d}.png"
            
            # save center crop RGB images
            img = cv2.imread(img_path)
            if self.resize == "resize":
                img = cv2.resize(img, (512, 512), interpolation = cv2.INTER_NEAREST)
            elif self.resize == "center_crop":
                img = img[img.shape[0] // 2 - w // 2 : img.shape[0] // 2 + w // 2, img.shape[1] // 2 - w // 2 : img.shape[1] // 2 + w // 2]
            cv2.imwrite(os.path.join(self.rgb, filename), img)

            # save segmentation labels in Nuscenes format
            semantic_mask, _ = self.nuim.get_segmentation(key_camera_token)
            if self.resize == "resize":
                semantic_mask = cv2.resize(semantic_mask, (512, 512), interpolation = cv2.INTER_NEAREST)
            elif self.resize == "center_crop":
                semantic_mask = semantic_mask[semantic_mask.shape[0] // 2 - w // 2 : semantic_mask.shape[0] // 2 + w // 2, semantic_mask.shape[1] // 2 - w // 2 : semantic_mask.shape[1] // 2 + w // 2]
            cv2.imwrite(os.path.join(self.seg_nuscenes, filename), semantic_mask)
            
            # clean up to avoid memory leaks
            del samples, key_camera_token, img, semantic_mask
    
    @staticmethod
    def map_nuscenes_to_carla(seg_nuscenes, seg_carla):
        '''
        Static method to map Nuscenes labels to Carla labels
        '''
        images = os.listdir(seg_nuscenes)
        for image in tqdm(images):
            img = cv2.imread(os.path.join(seg_nuscenes, image))
            img_copy = img.copy()
            for key, value in NUSCENES_CARLA_MAP.items():
                img_copy[img==key] = value
            cv2.imwrite(os.path.join(seg_carla, image), img_copy)   
          

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load VQA.')
    parser.add_argument('--data', type=str, choices=['train', 'val', 'test'], default='train')
    parser.add_argument('--dataroot', type=str, default='data/sets/nuimages')
    parser.add_argument('--save_folder', type=str, default="/content/train")
    parser.add_argument('--type', type=str, default='front')
    parser.add_argument('--sensor', type=str, default='CAM_FRONT')
    parser.add_argument('--resize', type=str, choices=['resize', 'center_crop'], default='resize')
    parser.add_argument('--num_images_to_save', type=int, default=7500)
    args = parser.parse_args()

    version = f"v1.0-{args.data}"
    create_dir(args.save_folder)
    
    nuscenes = NuimagesDataset(save_folder=args.save_folder, type=args.type, dataroot=args.dataroot, version=version, sensor=args.sensor, resize=args.resize, num_images_to_save=args.num_images_to_save)

    nuscenes.crate_dataset()
    print(f"{args.data} dataset collection done")
    
    # independent class mathod
    seg_nuscenes = os.path.join(args.save_folder, args.type, "seg_nuscenes")
    seg_carla = os.path.join(args.save_folder, args.type, "seg")
    NuimagesDataset.map_nuscenes_to_carla(seg_nuscenes, seg_carla)