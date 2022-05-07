import os

import torch
from albumentations import (Compose, HorizontalFlip, Normalize, RandomCrop,
                            Resize)
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms as T

from dataloaders.atdt_dataset import AtdtDataset
from dataloaders.carla_nuscenes_map import NUSCENES_CARLA_MAP


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

def get_front_dataloaders(cfg):
    
    torch.manual_seed(cfg.seed)

    nuscenes_files = get_clean_files_list(cfg.data.nuscenes_rgb_dir,
                                 cfg.data.nuscenes_seg_dir)

    rand_perm = torch.randperm(len(nuscenes_files))
    nuscenes_train_split = int(cfg.training.train_split * len(nuscenes_files))
    nuscenes_val_split = int(cfg.training.val_split * len(nuscenes_files))

    nuscenes_train_files = [nuscenes_files[i] for i in rand_perm[:nuscenes_train_split]]
    nuscenes_val_files = [nuscenes_files[i] for i in rand_perm[nuscenes_train_split: nuscenes_train_split + nuscenes_val_split]]
    
    carla_files = get_clean_files_list(cfg.data.carla_rgb_dir,
                                 cfg.data.carla_seg_dir)

    rand_perm = torch.randperm(len(carla_files))
    carla_train_split = int(cfg.training.train_split * len(carla_files))
    carla_val_split = int(cfg.training.val_split * len(carla_files))

    carla_train_files = [carla_files[i] for i in rand_perm[:carla_train_split]]
    carla_val_files = [carla_files[i] for i in rand_perm[carla_train_split: carla_train_split + carla_val_split]]

    # Do NOT include ToTensor and Normalize. These are done explicitly
    # on images.
    train_transforms = Compose([RandomCrop(cfg.training.crop_h,
                                           cfg.training.crop_w),
                                HorizontalFlip(p=0.5)],
                                additional_targets={'gt': 'mask'})

    val_transforms = Compose([Resize(cfg.training.crop_h,
                                     cfg.training.crop_w)],
                              additional_targets={'gt': 'mask'})

    nuscenes_labels_map = None
    carla_labels_map = None

    if cfg.data.need_labels_map:
        nuscenes_labels_map = NUSCENES_CARLA_MAP

    
    nuscenes_train_dataset = AtdtDataset(nuscenes_train_files,
                                image_dir=cfg.data.nuscenes_rgb_dir,
                                gt_dir=cfg.data.nuscenes_seg_dir,
                                transforms=train_transforms,
                                img_size=(cfg.training.crop_h,
                                          cfg.training.crop_w),
                                labels_map=nuscenes_labels_map)
    
    nuscenes_val_dataset = AtdtDataset(nuscenes_val_files,
                                image_dir=cfg.data.nuscenes_rgb_dir,
                                gt_dir=cfg.data.nuscenes_seg_dir,
                                transforms=val_transforms,
                                img_size=(cfg.training.crop_h,
                                        cfg.training.crop_w),
                                labels_map=nuscenes_labels_map)
    
    carla_train_dataset = AtdtDataset(carla_train_files,
                                image_dir=cfg.data.carla_rgb_dir,
                                gt_dir=cfg.data.carla_seg_dir,
                                transforms=train_transforms,
                                img_size=(cfg.training.crop_h,
                                          cfg.training.crop_w),
                                labels_map=carla_labels_map)
    
    carla_val_dataset = AtdtDataset(carla_val_files,
                                image_dir=cfg.data.carla_rgb_dir,
                                gt_dir=cfg.data.carla_seg_dir,
                                transforms=val_transforms,
                                img_size=(cfg.training.crop_h,
                                        cfg.training.crop_w),
                                labels_map=carla_labels_map)
    
    nuscenes_carla_train_dataset = ConcatDataset([nuscenes_train_dataset, carla_train_dataset])
    nuscenes_carla_val_dataset = ConcatDataset([nuscenes_val_dataset, carla_val_dataset])

    train_loader = DataLoader(nuscenes_carla_train_dataset,
                              batch_size=cfg.training.batch_train,
                              shuffle=True,
                              num_workers=cfg.training.n_workers)
    val_loader = DataLoader(nuscenes_carla_val_dataset,
                            batch_size=cfg.training.batch_val,
                            shuffle=False,
                            num_workers=cfg.training.n_workers)

    return train_loader, val_loader

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
    train_transforms = Compose([RandomCrop(cfg.training.crop_h,
                                           cfg.training.crop_w),
                                HorizontalFlip(p=0.5)],
                                additional_targets={'gt': 'mask'})

    val_transforms = Compose([Resize(cfg.training.crop_h,
                                     cfg.training.crop_w)],
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