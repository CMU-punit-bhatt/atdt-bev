import logging
import os
import random
from collections import OrderedDict
from ctypes import util

import cv2
import hydra
import numpy as np
import torch
import torch.optim as optim
from albumentations import Compose, HorizontalFlip, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import dataloaders.dataloader as dataloader
import utils.utils
from utils import ipm
from utils.losses import get_loss_fn
from utils.metrics import get_metrics
from utils.models import get_network, get_transfer


def get_lr(opt):

    return opt.param_groups[0]['lr']

def evaluate(N1,
             N2,
             G,
             loss_fn,
             dataset_dl,
             metrics=None,
             device=None,
             n1_transforms=None,
             n2_transforms=None):

    # set model to evaluation mode
    G.eval()
    metrics_results = {}

    if loss_fn is not None:
        running_loss = utils.RunningAverage()

    if metrics is not None:
        for metric_name, metric in metrics.items():
            metric.reset()

    with torch.no_grad():
        for (xb, _) in tqdm(dataset_dl):

            xb = xb.to(device)
            # yb = yb.to(device)

            encoding1 = N1(n1_transforms(xb))['out']
            encoding2 = N2(n2_transforms(xb))['out']

            output = G(encoding1)
            loss_b = loss_fn(output, encoding2)

            if loss_fn is not None:
                loss_b = loss_fn(output, encoding2)
                running_loss.update(loss_b.item())
            if metrics is not None:
                for metric_name, metric in metrics.items():
                    metric.add(output, encoding2)

    if metrics is not None:
        for metric_name, metric in metrics.items():
            metrics_results[metric_name] = metric.value()

    if loss_fn is not None:
        return running_loss(), metrics_results
    else:
        return None, metrics_results

def train_epoch(N1,
                N2,
                G,
                loss_fn,
                dataset_dl,
                opt=None,
                lr_scheduler=None,
                metrics=None,
                device=None,
                n1_transforms=None,
                n2_transforms=None):

    running_loss = utils.RunningAverage()

    if metrics is not None:
        for metric_name, metric in metrics.items():
            metric.reset()

    for (xb, _) in tqdm(dataset_dl):
        xb = xb.to(device)
        # yb = yb.to(device)

        encoding1 = N1(n1_transforms(xb))['out']
        encoding2 = N2(n2_transforms(xb))['out']

        output = G(encoding1)
        loss_b = loss_fn(output, encoding2)

        # print(encoding1.shape, encoding2.shape, output.shape)

        if opt is not None:
            opt.zero_grad()
            loss_b.backward()
            opt.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        running_loss.update(loss_b.item())

        if metrics is not None:
            for metric_name, metric in metrics.items():
                metric.add(output.detach(), encoding2)

    if metrics is not None:
        metrics_results = OrderedDict({})

        for metric_name, metric in metrics.items():
            metrics_results[metric_name] = metric.value()

        return running_loss(), metrics_results

    else:

        return running_loss(), None


def train_and_evaluate(N1,
                       N2,
                       G,
                       train_dataloader,
                       val_dataloader,
                       opt,
                       loss_fn,
                       metrics,
                       params,
                       lr_scheduler,
                       ckpt_dir,
                       ckpt_filename,
                       log_dir,
                       writer,
                       load_checkpoint=True,
                       device=None,
                       n1_train_transforms=None,
                       n1_val_transforms=None,
                       n2_train_transforms=None,
                       n2_val_transforms=None):

    ckpt_file_path = os.path.join(ckpt_dir, ckpt_filename)
    best_value = -float('inf')
    early_stopping = utils.EarlyStopping(patience=10, verbose=True)
    start_epoch = 0

    # batch_sample_train, batch_gt_train = next(iter(train_dataloader))
    # batch_sample_val, batch_gt_val = next(iter(val_dataloader))

    if load_checkpoint and os.path.exists(ckpt_file_path):
        G, opt, lr_scheduler, start_epoch, best_value = \
            utils.load_checkpoint(G,
                                  opt,
                                  lr_scheduler,
                                  start_epoch,
                                  False,
                                  best_value,
                                  ckpt_dir,
                                  ckpt_filename)
        print("=> loaded checkpoint form {} (epoch {})".format(
            ckpt_file_path, start_epoch))
    else:
        print("=> Initializing from scratch")

    for epoch in range(start_epoch, params.n_epochs):
        # Run one epoch
        current_lr = get_lr(opt)
        logging.info('Epoch {}/{}, current lr={}'.format(epoch,
                                                         params.n_epochs - 1,
                                                         current_lr))
        writer.add_scalar('Learning_rate', current_lr, epoch)

        G.train()
        train_loss, train_metrics = train_epoch(N1,
                                                N2,
                                                G,
                                                loss_fn,
                                                train_dataloader,
                                                opt,
                                                lr_scheduler,
                                                metrics,
                                                params,
                                                device=device,
                                                n1_transforms=n1_train_transforms,
                                                n2_transforms=n2_train_transforms)

        # Evaluate for one epoch on validation set
        val_loss, val_metrics = evaluate(N1,
                                         N2,
                                         G,
                                         loss_fn,
                                         val_dataloader,
                                         metrics=metrics,
                                         device=device,
                                         n1_transforms=n1_val_transforms,
                                         n2_transforms=n2_val_transforms)

        writer.add_scalars('Loss', {
            'Training': train_loss,
            'Validation': val_loss,
        }, epoch)

        for (train_metric_name, train_metric_results), \
            (val_metric_name, val_metric_results) in zip(train_metrics.items(),
                                                         val_metrics.items()):
            writer.add_scalars(train_metric_name, {
                'Training': train_metric_results[0],
                'Validation': val_metric_results[0],
            }, epoch)

        current_value = list(val_metrics.values())[0][0]
        is_best = current_value >= best_value

        # If best_eval, best_save_path
        if is_best:
            logging.info("- Found new best accuracy")
            best_value = current_value
            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                log_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': G.state_dict(),
                               'optim_dict': opt.state_dict(),
                               'scheduler_dict': lr_scheduler.state_dict(),
                               'best_value': best_value},
                               is_best=is_best,
                               ckpt_dir=ckpt_dir,
                               filename=ckpt_filename)

        logging.info("\ntrain loss: %.3f, val loss: %.3f" %
                     (train_loss, val_loss))
        for (train_metric_name, train_metric_results), \
            (val_metric_name, val_metric_results) in zip(train_metrics.items(),
                                                         val_metrics.items()):
            logging.info("train %s: %.3f, val %s: %.3f" % (train_metric_name,
                                                           train_metric_results[0],
                                                           val_metric_name,
                                                           val_metric_results[0]))

        logging.info("-"*20)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            logging.info("Early stopping")
            break

@hydra.main(config_path='./configs', config_name='train_transfer')
def main(cfg: DictConfig):

    params = cfg.training
    ckpt_filename = "G.pt"
    log_dir = os.path.join(cfg.logging.log_dir, f'{cfg.model_name}/{cfg.exp}/')
    ckpt_dir = os.path.join(cfg.logging.ckpt_dir, f'{cfg.model_name}/{cfg.exp}')
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
    train_loader, val_loader = dataloader.get_g_dataloaders(cfg)

    logging.info("- done.")

    # Adding transforms - Include ToTensor and normalize.
    n1_train_transforms = Compose([Resize(cfg.training.crop_h,
                                          cfg.training.crop_w,
                                          interpolation=cv2.INTER_NEAREST),
                                   HorizontalFlip(p=0.5),
                                   Normalize(),
                                   ToTensorV2()],
                                   additional_targets={'gt': 'mask'})

    n1_val_transforms = Compose([Resize(cfg.training.crop_h,
                                        cfg.training.crop_w,
                                        interpolation=cv2.INTER_NEAREST),
                                 Normalize(),
                                 ToTensorV2()],
                                 additional_targets={'gt': 'mask'})

    n2_train_transforms = Compose([ipm.create_lambda_transform(),
                                   Resize(cfg.training.crop_h,
                                          cfg.training.crop_w,
                                          interpolation=cv2.INTER_NEAREST),
                                   HorizontalFlip(p=0.5),
                                   Normalize(),
                                   ToTensorV2()],
                                   additional_targets={'gt': 'mask'})

    n2_val_transforms = Compose([ipm.create_lambda_transform(),
                                 Resize(cfg.training.crop_h,
                                        cfg.training.crop_w,
                                        interpolation=cv2.INTER_NEAREST),
                                 Normalize(),
                                 ToTensorV2()],
                                 additional_targets={'gt': 'mask'})

    # Define the model and optimizer
    N1 = get_network(params).to(device)
    N2 = get_network(params).to(device)

    # Loading and freezing N1 and N2.
    N1 = utils.load_checkpoint(N1,
                               ckpt_dir=cfg.data.n1_ckpt_dir,
                               filename=cfg.data.n1_ckpt_name)[0]
    N2 = utils.load_checkpoint(N2,
                               ckpt_dir=cfg.data.n2_ckpt_dir,
                               filename=cfg.data.n2_ckpt_name)[0]
    N1 = N1.backbone.to(device)
    N2 = N2.backbone.to(device)

    for param in N1.parameters():
        param.requires_grad = False
    for param in N2.parameters():
        param.requires_grad = False

    G = get_transfer(params).to(device)

    opt = optim.Adam(G.parameters(), lr=params.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt,
                                                   cfg.training.lr_step_epochs * \
                                                        len(train_loader),
                                                   gamma=cfg.training.lr_gamma)

    # fetch loss function and metrics
    loss_fn = get_loss_fn(params)

    metrics = OrderedDict({})
    for metric in params.metrics:
        metrics[metric] = get_metrics(metric, params)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.n_epochs))
    train_and_evaluate(N1,
                       N2,
                       G,
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
                       load_checkpoint=False,
                       device=device,
                       n1_train_transforms=n1_train_transforms,
                       n1_val_transforms=n1_val_transforms,
                       n2_train_transforms=n2_train_transforms,
                       n2_val_transforms=n2_val_transforms)

if __name__ == '__main__':
    main()
