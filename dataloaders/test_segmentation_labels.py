import cv2
import os
from tqdm import tqdm
import torch
import argparse
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


'''
Dataloader to fetch all segmentation labels
'''
class Dataset(Dataset):
    def __init__(self, image_dir):
        self._image_dir = image_dir
        self._images = os.listdir(self._image_dir)
        
    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        image_path = os.path.join(self._image_dir, self._images[idx])
        image = cv2.imread(image_path, 0)
        return image


def flatten_to_list(object):
    '''
    Flatten a nested list/tuple/set
    '''
    gather = []
    for item in object:
        if isinstance(item, (list, tuple, set)):
            gather.extend(flatten_to_list(item))            
        else:
            gather.append(item)
    return set(gather)


def test(img_path):
    '''
    Get all the unique labels present in an image
    '''
    classes = set()
    train_dataset = Dataset(img_path)
    train_dataset_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
    for batch in tqdm(train_dataset_loader):
        flatten = torch.unique(torch.flatten(batch.cuda()))
        classes.add(tuple(flatten.cpu().numpy()))
    return flatten_to_list(classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load VQA.')
    parser.add_argument('--img_path', type=str, default='/content/train/front/seg')
    args = parser.parse_args()
    classes = test(args.img_path)
    print(classes)