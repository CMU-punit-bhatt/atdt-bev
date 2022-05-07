import os
import hydra
from tqdm import tqdm
import dataloaders.dataloader as dataloader
from collections import OrderedDict
from omegaconf import DictConfig


@hydra.main(config_path='./configs', config_name='train_front')
def main(cfg: DictConfig):

    # fetch dataloaders
    # NOTE: Check with batch size 1 in the config file.
    
    train_loader, val_loader = dataloader.get_front_dataloaders(cfg)

    num_train_images, num_val_images = len(train_loader), len(val_loader) 
    print(f"Size of train loader: {num_train_images}\nSize of val loader: {num_val_images}")
    
    num_images_nuscenes, num_images_carla = len(os.listdir(cfg.data.nuscenes_rgb_dir)), len(os.listdir(cfg.data.carla_seg_dir))
    print(f"Size of nuscenes folder: {num_images_nuscenes}\nSize of carla folder: {num_images_carla}")
    
    assert (num_train_images + num_val_images) == (num_images_nuscenes + num_images_carla), "Combining dataset failed!"

if __name__ == '__main__':
    main()