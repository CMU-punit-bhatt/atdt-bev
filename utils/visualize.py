import numpy as np
import torch
import torchvision
import matplotlib as plt
from PIL import Image
import os
import utils
from utils.utils import load_checkpoint
from wandb import visualize
from dataloaders.carla_nuscenes_map import CARLA_CLASSES_NAME_TO_RGB, NUSCENES_CARLA_MAP
import dataloaders.dataloader as dataloader
import cv2
import hydra
from collections import OrderedDict
from omegaconf import DictConfig
from hydra import compose, initialize
from omegaconf import OmegaConf
from utils.models import get_network
import argparse
from tqdm import tqdm

# reference: https://datascience.stackexchange.com/questions/40637/how-to-visualize-image-segmentation-results

def visualise(model, input, save_path, channels_dim=1):
    '''
    args:
    model: model for inference
    input: input image to run inference
    save_path: list of save paths to each of the output image
    '''

    output = model(input.cuda())
    output = output["out"]
    # assuming output shape of B x N x H x W
    segmented_image = np.zeros((output.shape[0], output.shape[-2], output.shape[-1], 3))  

    # assuming N channels in dim=1
    output = torch.argmax(output, dim=channels_dim)
    output = output.detach().cpu().numpy()

    for k, v in CARLA_CLASSES_NAME_TO_RGB.items():
        segmented_image[output==k] = v

    for i, path in enumerate(save_path):
        out_img = segmented_image[i]
        out_img = out_img.astype(np.uint8)
        out_img = Image.fromarray(out_img).convert('RGB')
        out_img.save(path)
        
        
def visualise_gt(img_paths, save_paths):
    '''
    args:
    model: model for inference
    input: input image to run inference
    save_path: list of save paths to each of the output image
    '''
    
    for img_path, path in tqdm(zip(img_paths, save_paths)):
        img = cv2.imread(img_path)
        carla_img = np.zeros_like(img)
         
        for k, v in NUSCENES_CARLA_MAP.items():
            carla_img[img == k] = v
        
        assert np.all(carla_img[:, :, 0]==carla_img[:, :, 1])
        assert np.all(carla_img[:, :, 0]==carla_img[:, :, 2])
        assert np.all(carla_img[:, :, 2]==carla_img[:, :, 1])
        carla_img = carla_img[:, :, 2]
        
        segmented_image = np.zeros_like(img)  
        
        for k, v in CARLA_CLASSES_NAME_TO_RGB.items():
            segmented_image[carla_img==k] = v
        
        # cv2.imwrite(path, out_img)
        out_img = Image.fromarray(segmented_image).convert('RGB')
        out_img.save(path)



def unit_test1():
    batch_image = None
    save_path = []
    model = lambda x: x
    
    save_folder = "test"
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    
    for i in range(23):
        img = torch.ones(1, 1, 224, 224) * i
        if batch_image is None:
            batch_image = img
        else:
            batch_image = torch.cat([batch_image, img], dim=0)
        path = os.path.join(save_folder, f"image_{i}.png")
        save_path.append(path)
        
    visualise(model, batch_image, save_path)
  

@hydra.main(config_path='./configs', config_name='train_front')
def main_n1(cfg: DictConfig):
    
    # TODO: figure out passing args for hydra functions
    ckpt_filename = "/home/adithyas/atdt/logs/front_syn/0/checkpoint_front.pt"
    img_paths = ["/home/adithyas/atdt/content/train_full/front/rgb/000127.png", "/home/adithyas/atdt/content/train_full/front/rgb/000560.png"]
    save_paths = ["/home/adithyas/atdt/results/000127.png", "/home/adithyas/atdt/results/000560.png"]
    
    params = cfg.training
    model = get_network(params).to(torch.device("cuda"))
    best_value = -float('inf')
    model, _, _, _, best_value = load_checkpoint(model,
                                  None,
                                  None,
                                  0,
                                  False,
                                  best_value,
                                  ".",
                                  ckpt_filename)
    model.eval()
        
    input = None
    
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        img = Image.fromarray(img).convert('RGB')
        toTensor = torchvision.transforms.Compose([torchvision.transforms.Resize((512, 512)), torchvision.transforms.ToTensor()]) 
        img = toTensor(img)
        if input is None:
            input = img.unsqueeze(0).cuda()
        else:
            input = torch.cat([input, img.unsqueeze(0).cuda()], dim=0)
        
    visualise(model, input, save_paths)

@hydra.main(config_path='./configs', config_name='train_bev')
def main_n2(cfg: DictConfig, ckpt_filename, img_paths, save_paths):
    params = cfg.training
    model = get_network(params).to(torch.device("cuda"))
    best_value = -float('inf')
    model, _, _, _, best_value = load_checkpoint(model,
                                  None,
                                  None,
                                  0,
                                  False,
                                  best_value,
                                  ".",
                                  ckpt_filename)
    model.eval()
    
    input = None
    
    for img_path in tqdm(img_paths):
        img = cv2.imread(img_path)
        img = Image.fromarray(img).convert('RGB')
        toTensor = torchvision.transforms.Compose([torchvision.transforms.Resize((512, 512)), torchvision.transforms.ToTensor()]) 
        img = toTensor(img)
        if input is None:
            input = img.unsqueeze(0).cuda()
        else:
            input = torch.cat([input, img.unsqueeze(0).cuda()], dim=0)
        
    visualise(model, input, save_paths)

if __name__ == '__main__':
    # TODO: add argeparse
    parser = argparse.ArgumentParser(description='Load labels.')
    parser.add_argument('--ckpt_filename', type=str, default="/home/adithyas/atdt/logs/front_syn/0/checkpoint_front.pt")
    parser.add_argument('--img_paths', type=str, default=["/home/adithyas/atdt/content/train_full/front/seg/000127.png", "/home/adithyas/atdt/content/train_full/front/seg/000560.png"])
    parser.add_argument('--save_paths', type=str, default=["/home/adithyas/atdt/results/000127.png", "/home/adithyas/atdt/results/000560.png"])
    args = parser.parse_args()
    # main_n1()
    # main_n2()
    visualise_gt(args.img_paths, args.save_paths)

    
    