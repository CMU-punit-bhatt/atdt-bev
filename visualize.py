import numpy as np
import torch
import torchvision
import matplotlib as plt
from PIL import Image
import os
import utils
from wandb import visualize
from carla_nuscenes_map import CARLA_CLASSES_NAME_TO_RGB
import dataloaders.dataloader as dataloader
import cv2
import hydra
from collections import OrderedDict
from omegaconf import DictConfig
from models import get_network

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

    for key in CARLA_CLASSES_NAME_TO_RGB:
        segmented_image[output==key] = CARLA_CLASSES_NAME_TO_RGB[key]

    for i, path in enumerate(save_path):
        out_img = segmented_image[i]
        out_img = out_img.astype(np.uint8)
        out_img = Image.fromarray(out_img).convert('RGB')
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
  

# @hydra.main(config_path='./configs', config_name='train_front')
# def main(cfg: DictConfig):
#     params = cfg.training
#     ckpt_filename = "/home/adithyas/atdt/n1_model_best.pt"
#     model = get_network(params).to(torch.device("cuda"))
#     best_value = -float('inf')
#     model, _, _, _, best_value = utils.load_checkpoint(model,
#                                   None,
#                                   None,
#                                   0,
#                                   False,
#                                   best_value,
#                                   ".",
#                                   ckpt_filename)
#     model.eval()
#     file_list = ["000050", "000052", "000053", "000088", "002180", "002110"]
#     img_format = "/home/adithyas/atdt/{}_rgb.png"
#     save_format = "/home/adithyas/atdt/{}_result.png"
#     img_paths = [img_format.format(file) for file in file_list]
#     save_paths = [save_format.format(file) for file in file_list]
#     input = None
    
#     for img_path, save_path in zip(img_paths, save_paths):
#         img = cv2.imread(img_path)
#         img = Image.fromarray(img).convert('RGB')
#         toTensor = torchvision.transforms.Compose([torchvision.transforms.Resize((512, 512)), torchvision.transforms.ToTensor()]) 
#         img = toTensor(img)
#         if input is None:
#             input = img.unsqueeze(0).cuda()
#         else:
#             input = torch.cat([input, img.unsqueeze(0).cuda()], dim=0)
        
#     visualise(model, input, save_paths)

@hydra.main(config_path='./configs', config_name='train_bev')
def main(cfg: DictConfig):
    params = cfg.training
    ckpt_filename = "/home/adithyas/atdt/n2_model_best.pt"
    model = get_network(params).to(torch.device("cuda"))
    best_value = -float('inf')
    model, _, _, _, best_value = utils.load_checkpoint(model,
                                  None,
                                  None,
                                  0,
                                  False,
                                  best_value,
                                  ".",
                                  ckpt_filename)
    model.eval()
    file_list = ["000050", "000052", "000053", "000088", "002180", "002110"]
    img_format = "/home/adithyas/atdt/{}_rgb.png"
    save_format = "/home/adithyas/atdt/{}_bev.png"
    img_paths = [img_format.format(file) for file in file_list]
    save_paths = [save_format.format(file) for file in file_list]
    input = None
    
    for img_path, save_path in zip(img_paths, save_paths):
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
    main()

    
    