import numpy as np
import torch
import torchvision
import matplotlib as plt
from PIL import Image
import os
from carla_nuscenes_map import CARLA_CLASSES_NAME_TO_RGB

# reference: https://datascience.stackexchange.com/questions/40637/how-to-visualize-image-segmentation-results

def visualise(model, input, save_path, channels_dim=1):
    '''
    args:
    model: model for inference
    input: input image to run inference
    save_path: list of save paths to each of the output image
    '''
    output = model(input.cuda())
    
    # assuming output shape of B x N x H x W
    segmented_image = np.zeros((output.shape[0], output.shape[-2], output.shape[-1], 3))  
    
    # assuming N channels in dim=1
    output = torch.amax(output, dim=channels_dim)
    output = output.detach().cpu().numpy()
    
    for key in CARLA_CLASSES_NAME_TO_RGB:
        segmented_image[output==key] = CARLA_CLASSES_NAME_TO_RGB[key]
    
    for i, path in enumerate(save_path):
        out_img = segmented_image[i]
        out_img = out_img.astype(np.uint8)
        out_img = Image.fromarray(out_img).convert('RGB')
        out_img.save(path)

# Unit Test script
if __name__ == "__main__":
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
    
    