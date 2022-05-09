from collections import OrderedDict

import cv2
import hydra
import torch
from albumentations import Compose, HorizontalFlip, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm

from dataloaders.dataloader import get_test_dataloader
from utils import ipm
from utils.losses import get_loss_fn
from utils.metrics import get_metrics
from utils.models import get_adaptive_network, get_network, get_transfer
from utils.utils import RunningAverage, load_checkpoint


def get_adaptive_net(cfg):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    params = cfg.training

    # Define the model and optimizer
    N1 = get_network(params).to(device)
    N2 = get_network(params).to(device)
    G = get_transfer(params).to(device)

    # Loading and freezing N1 and N2.
    N1 = load_checkpoint(N1,
                         ckpt_dir=cfg.data.ckpt_dir,
                         filename=cfg.data.n1_ckpt_name)[0]
    N2 = load_checkpoint(N2,
                         ckpt_dir=cfg.data.ckpt_dir,
                         filename=cfg.data.n2_ckpt_name)[0]
    G = load_checkpoint(G,
                        ckpt_dir=cfg.data.ckpt_dir,
                        filename=cfg.data.g_ckpt_name)[0]

    return get_adaptive_network(N1.backbone, G, N2.classifier)


@hydra.main(config_path='./configs', config_name='eval_adaptive')
def main(cfg: DictConfig):

    params = cfg.training

    n1_transforms = Compose([Resize(cfg.training.crop_h,
                                    cfg.training.crop_w,
                                    interpolation=cv2.INTER_NEAREST),
                             Normalize(),
                             ToTensorV2()],
                             additional_targets={'gt': 'mask'})

    n2_transforms = Compose([ipm.create_lambda_transform(),
                             Resize(cfg.training.crop_h,
                                    cfg.training.crop_w,
                                    interpolation=cv2.INTER_NEAREST),
                             Normalize(),
                             ToTensorV2()],
                             additional_targets={'gt': 'mask'})

    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() and \
        cfg.training.use_cuda else 'cpu')

    model_adapt = get_adaptive_net(cfg).to(device)
    model_adapt.eval()

    model_n2 = get_network(params).to(device)
    model_n2 = load_checkpoint(model_n2,
                               ckpt_dir=cfg.data.ckpt_dir,
                               filename=cfg.data.n2_ckpt_name)[0]

    test_loader = get_test_dataloader(cfg)

    # fetch loss function and metrics
    loss_fn = get_loss_fn(params)
    running_loss_adapt = RunningAverage()
    running_loss_n2 = RunningAverage()

    metrics_adapt = OrderedDict({})
    metrics_n2 = OrderedDict({})
    for metric in params.metrics:
        metrics_adapt[metric] = get_metrics(metric, params)
        metrics_n2[metric] = get_metrics(metric, params)

    for (xb, yb) in tqdm(test_loader):

        xb = xb.to(device)
        yb = yb.to(device)

        xb_adapt = []
        xb_n2 = []

        for i in range(xb.size(0)):
            xb_adapt.append(n1_transforms(xb[i]))
            xb_n2.append(n2_transforms(xb[i]))

        xb_adapt = torch.stack(xb_adapt).to(device)
        xb_n2 = torch.stack(xb_n2).to(device)

        out_adapt = model_adapt(xb_adapt)['out']
        out_n2 = model_adapt(xb_n2)['out']

        if loss_fn is not None:
            loss_adapt = loss_fn(out_adapt, yb)
            loss_n2 = loss_fn(out_n2, yb)
            running_loss_adapt.update(loss_adapt.item())
            running_loss_n2.update(loss_n2.item())
        if metrics_adapt is not None:
            for _, metric in metrics_adapt.items():
                metric.add(out_adapt, yb)
        if metrics_n2 is not None:
            for _, metric in metrics_n2.items():
                metric.add(out_n2, yb)

        # Need to add some changes for saving or viz here.

    for metric_name, metric in metrics_n2.items():
        print(f'N2: {metric_name}: {metric}')

    for metric_name, metric in metrics_adapt.items():
        print(f'Adaptive: {metric_name}: {metric}')

if __name__ == '__main__':
    main()
