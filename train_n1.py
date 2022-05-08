import logging
import os
import random
from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import dataloaders.dataloader as dataloader
import utils.utils as utils
from utils.losses import get_loss_fn
from utils.metrics import get_metrics
from utils.models import get_network
from utils.train_utils import train_and_evaluate

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

@hydra.main(config_path='./configs', config_name='train_front')
def main(cfg: DictConfig):

    params = cfg.training
    ckpt_filename = "checkpoint_front.pt"
    log_dir = os.path.join(cfg.logging.log_dir, f'{cfg.model_name}/{cfg.exp}/')
    ckpt_dir = os.path.join(cfg.logging.log_dir, f'{cfg.model_name}/{cfg.exp}')
    
    create_dir(log_dir)
    create_dir(ckpt_dir)
    
    writer = SummaryWriter(log_dir)

    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() and \
        cfg.training.use_cuda else 'cpu')

    # Set the random seed for reproducible experiments
    seed = cfg.seed

    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Set the logger
    if not os.path.exists(log_dir):
        print("Making log directory {}".format(log_dir))
        os.mkdir(log_dir)
    utils.set_logger(os.path.join(log_dir, "train.log"))

    # Create the input data pipeline
    logging.info("Loading the datasets...")

    # fetch dataloaders
    train_loader, val_loader = dataloader.get_front_dataloaders(cfg)

    logging.info("- done.")

    # Define the model and optimizer
    model = get_network(params).to(device)
    opt = optim.Adam(model.parameters(), lr=params.lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(opt,
                                                       max_lr=params.lr,
                                                       steps_per_epoch=\
                                                        len(train_loader),
                                                        epochs=params.n_epochs,
                                                        div_factor=20)

    # fetch loss function and metrics
    loss_fn = get_loss_fn(params)

    metrics = OrderedDict({})
    for metric in params.metrics:
        metrics[metric] = get_metrics(metric, params)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.n_epochs))
    train_and_evaluate(model,
                       train_loader,
                       val_loader,
                       opt,
                       loss_fn,
                       metrics,
                       params,
                       lr_scheduler,
                       ckpt_dir,
                       ckpt_filename,
                       log_dir,
                       writer,
                       device=device)

if __name__ == '__main__':
    main()