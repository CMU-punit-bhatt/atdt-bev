import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class AtdtDataset(Dataset):

    def __init__(self,
                 file_names,
                 image_dir='./data/syn/front/rgb/',
                 gt_dir='./data/syn/front/seg/',
                 transforms=None,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 img_ext='png',
                 gt_ext='png',
                 img_size=(512, 512),
                 labels_map=None):

        super(AtdtDataset, self).__init__()

        assert os.path.exists(image_dir)
        assert os.path.exists(gt_dir)

        self.img_dir = image_dir
        self.gt_dir = gt_dir
        self.mean = mean
        self.std = std
        self.img_ext = img_ext
        self.gt_ext = gt_ext
        self.img_size = img_size
        self.labels_map = labels_map

        self.base_transforms = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=self.mean, std=self.std),
            ])

        self.transforms = transforms

        if transforms is None:

            self.transforms = T.Compose([
                T.Resize(self.img_size),
            ])

        self.file_names = file_names

    def re_normalize(self, img):
        img_r = img.clone()

        for c, (mean_c, std_c) in enumerate(zip(self.mean, self.std)):
            img_r[c] *= std_c
            img_r[c] += mean_c

        return img_r

    def depth2color(self, depth_np):
        depth_np[depth_np > self.treshold] = self.treshold
        depth_np = (depth_np-depth_np.min())/(depth_np.max()-depth_np.min())
        indexes = np.array(depth_np*255, dtype=np.int32)
        color_depth = self.colors_depth[indexes]
        return color_depth

    def encode_image_train_id(self, mask):
        mask = np.array(mask)
        mask_copy = mask.copy()
        for k, v in self.id_to_trainId.items():
            mask_copy[mask == k] = v
        return mask_copy

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir,
                                '.'.join([self.file_names[idx], self.img_ext]))
        gt_path = os.path.join(self.gt_dir,
                               '.'.join([self.file_names[idx], self.gt_ext]))

        image = Image.open(img_path).convert('RGB')
        gt = cv2.imread(gt_path).astype(np.uint8)

        transformed = self.transforms(image=np.array(image), gt=np.array(gt))
        image = transformed['image']
        gt = transformed['gt']

        image = self.base_transforms(image)

        # TODO: Just taking first channel. Is that enough?
        gt = torch.from_numpy(np.array(gt))[..., 2]

        if self.labels_map is not None:

            assert type(self.labels_map) == dict

            mapped_gt = torch.zeros_like(gt)

            for k, v in self.labels_map.items():
                mapped_gt[gt == k] = v

            gt = mapped_gt

        return image, gt.long()

    def __len__(self):
        return len(self.file_names)
