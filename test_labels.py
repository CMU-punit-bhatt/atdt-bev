import cv2
import numpy as np
import os
from tqdm import tqdm
import torch

from torch.utils.data import Dataset
import torchvision

from torch.utils.data import DataLoader

from carla_nuscenes_map import NUSCENES_CARLA_MAP

class Dataset(Dataset):
    def __init__(self, image_dir):
        self._image_dir = image_dir
        self._images = os.listdir(self._image_dir)
        
    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image_path = os.path.join(self._image_dir, self._images[idx])
        image = cv2.imread(image_path)
        return image


def test(img_path):
    
    classes = set()
    train_dataset = Dataset(img_path)
    train_dataset_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
    for i, batch in enumerate(train_dataset_loader):
        print(f"Batch: {i}")
        flatten = torch.unique(torch.flatten(batch.cuda()))
        classes.add(tuple(flatten.cpu().numpy()))
    return classes

def test_mapping(img1, img2):
    img1 = img1.flatten()
    img2 = img2.flatten()
    for i1, i2 in zip(img1, img2):
        assert i1 == NUSCENES_CARLA_MAP(i2), "Not matching!"

if __name__ == "__main__":

    # img_path = '/home/adithyas/atdt/data/nuImages/mini/front/seg'
    # classes = test(img_path)
    # print(classes)
    
    img_path1 = '/home/adithyas/atdt/data/nuImages/mini/front/seg/000001.png' 
    img1 = cv2.imread(img_path1)
    img_path2 = '/home/adithyas/atdt/data/nuImages/mini/front/seg_viz/000001.png' 
    img2 = cv2.imread(img_path2)
    
    test_mapping(img1, img2)