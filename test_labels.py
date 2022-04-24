import cv2
import numpy as np
import os
from tqdm import tqdm
import torch

from torch.utils.data import Dataset
import torchvision

from torch.utils.data import DataLoader

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

if __name__ == "__main__":
    img_path = '/home/adithyas/atdt/data/nuImages/dataset/seg' 
    classes = test(img_path)
    print(classes)