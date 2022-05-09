from configparser import Interpolation
import os

import cv2
import torch
from albumentations import (Compose, HorizontalFlip, Normalize, RandomCrop,
                            Resize)
from torch.utils.data import DataLoader
from torchvision import transforms as T

from dataloaders.atdt_dataset import AtdtDataset
from dataloaders.carla_nuscenes_map import NUSCENES_CARLA_MAP
from utils import ipm


def get_clean_files_list(img_dir, gt_dir):

    # Removing missing and corrupt files.

    assert os.path.exists(img_dir)
    assert os.path.exists(gt_dir)

    img_files = os.listdir(img_dir)
    img_ext = img_files[0].split('.')[-1]
    gt_files = os.listdir(gt_dir)
    gt_ext = gt_files[0].split('.')[-1]

    img_files = [f.split('.')[0] for f in img_files]
    gt_files = [f.split('.')[0] for f in gt_files]

    to_remove = []

    for f in img_files:
        if f not in gt_files:
            to_remove.append(f)
        elif os.path.getsize(os.path.join(img_dir, f + '.' + img_ext)) == 0:
            to_remove.append(f)

    for f in gt_files:
        if f not in img_files:
            to_remove.append(f)
        elif os.path.getsize(os.path.join(gt_dir, f + '.' + gt_ext)) == 0:
            to_remove.append(f)

    comb_files = list(set(img_files) | set(gt_files))
    to_remove = list(set(to_remove))

    for f in to_remove:
        comb_files.remove(f)

    return comb_files

def get_bev_dataloaders(cfg):

    torch.manual_seed(cfg.seed)

    files = get_clean_files_list(cfg.data.front_rgb_dir,
                                 cfg.data.bev_seg_dir)

    rand_perm = torch.randperm(len(files))
    train_split = int(cfg.training.train_split * len(files))
    val_split = int(cfg.training.val_split * len(files))

    train_files = [files[i] for i in rand_perm[:train_split]]
    val_files = [files[i] for i in rand_perm[train_split: train_split + val_split]]

    # Do NOT include ToTensor and Normalize. These are done explicitly
    # on images.
    train_transforms = Compose([ipm.create_lambda_transform(),
                                Resize(cfg.training.crop_h,
                                       cfg.training.crop_w,
                                       interpolation=cv2.INTER_NEAREST),
                                HorizontalFlip(p=0.5)],
                                additional_targets={'gt': 'mask'})

    val_transforms = Compose([ipm.create_lambda_transform(),
                              Resize(cfg.training.crop_h,
                                     cfg.training.crop_w,
                                     interpolation=cv2.INTER_NEAREST)],
                              additional_targets={'gt': 'mask'})

    labels_map = None

    if cfg.data.need_labels_map:
        labels_map = NUSCENES_CARLA_MAP

    train_dataset = AtdtDataset(train_files,
                                image_dir=cfg.data.front_rgb_dir,
                                gt_dir=cfg.data.bev_seg_dir,
                                transforms=train_transforms,
                                img_size=(cfg.training.crop_h,
                                          cfg.training.crop_w),
                                labels_map=labels_map)
    val_dataset = AtdtDataset(val_files,
                              image_dir=cfg.data.front_rgb_dir,
                              gt_dir=cfg.data.bev_seg_dir,
                              transforms=val_transforms,
                              img_size=(cfg.training.crop_h,
                                        cfg.training.crop_w),
                              labels_map=labels_map)

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.training.batch_train,
                              shuffle=True,
                              num_workers=cfg.training.n_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.training.batch_val,
                            shuffle=False,
                            num_workers=cfg.training.n_workers)

    return train_loader, val_loader

def get_g_dataloaders(cfg):

    torch.manual_seed(cfg.seed)

    files = get_clean_files_list(cfg.data.front_rgb_dir,
                                 cfg.data.bev_seg_dir)

    rand_perm = torch.randperm(len(files))
    train_split = int(cfg.training.train_split * len(files))
    val_split = int(cfg.training.val_split * len(files))

    train_files = [files[i] for i in rand_perm[:train_split]]
    val_files = [files[i] for i in rand_perm[train_split: train_split + val_split]]

    labels_map = None

    if cfg.data.need_labels_map:
        labels_map = NUSCENES_CARLA_MAP

    train_dataset = AtdtDataset(train_files,
                                image_dir=cfg.data.front_rgb_dir,
                                gt_dir=cfg.data.bev_seg_dir,
                                transforms=None,
                                img_size=(cfg.training.crop_h,
                                          cfg.training.crop_w),
                                labels_map=labels_map,
                                use_base_transforms=False)
    val_dataset = AtdtDataset(val_files,
                              image_dir=cfg.data.front_rgb_dir,
                              gt_dir=cfg.data.bev_seg_dir,
                              transforms=None,
                              img_size=(cfg.training.crop_h,
                                        cfg.training.crop_w),
                              labels_map=labels_map,
                              use_base_transforms=False)

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.training.batch_train,
                              shuffle=True,
                              num_workers=cfg.training.n_workers)
    val_loader = DataLoader(val_dataset,
                            batch_size=cfg.training.batch_val,
                            shuffle=False,
                            num_workers=cfg.training.n_workers)

    return train_loader, val_loader

def get_test_dataloader(cfg):

    torch.manual_seed(cfg.seed)

    files = get_clean_files_list(cfg.data.test_front_rgb_dir,
                                 cfg.data.test_bev_seg_dir)

    labels_map = None

    if cfg.data.need_labels_map:
        labels_map = NUSCENES_CARLA_MAP

    test_dataset = AtdtDataset(files,
                                image_dir=cfg.data.test_front_rgb_dir,
                                gt_dir=cfg.data.test_bev_seg_dir,
                                transforms=None,
                                img_size=(cfg.training.crop_h,
                                          cfg.training.crop_w),
                                labels_map=labels_map,
                                use_base_transforms=False)

    test_loader = DataLoader(test_dataset,
                             batch_size=cfg.training.batch_test,
                             shuffle=False,
                             num_workers=cfg.training.n_workers)

    return test_loader