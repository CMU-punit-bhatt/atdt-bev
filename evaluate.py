"""Evaluates the model"""

import hydra
import os
import random
import numpy as np
import torch
import utils
from models import get_network
from tqdm import tqdm
import dataloaders.dataloader as dataloader
from losses import get_loss_fn
from metrics import get_metrics
from omegaconf import DictConfig

def evaluate(model, loss_fn, dataset_dl, metrics=None, device=None):

    # set model to evaluation mode
    model.eval()
    metrics_results = {}

    if loss_fn is not None:
        running_loss = utils.RunningAverage()

    if metrics is not None:
        for metric_name, metric in metrics.items():
            metric.reset()

    with torch.no_grad():
        for (xb, yb) in tqdm(dataset_dl):
            xb = xb.to(device)
            yb = yb.to(device)
            output = model(xb)['out']

            if loss_fn is not None:
                loss_b = loss_fn(output, yb)
                running_loss.update(loss_b.item())
            if metrics is not None:
                for metric_name, metric in metrics.items():
                    metric.add(output, yb)

    if metrics is not None:
        for metric_name, metric in metrics.items():
            metrics_results[metric_name] = metric.value()

    if loss_fn is not None:
        return running_loss(), metrics_results
    else:
        return None, metrics_results

@hydra.main(config_path='./configs', config_name='train_bev')
def main(cfg: DictConfig):
    """
        Evaluate the model on the test set.
    """
    params = cfg.training
    ckpt_filename = "checkpoint.pt"
    ckpt_dir = cfg.logging.ckpt_dir.rtrim('/') + f'/{cfg.model_name}/{cfg.exp}'

    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() and \
        cfg.training.use_cuda else 'cpu')

    # Set the random seed for reproducible experiments
    seed = cfg.seed

    # Set the random seed for reproducible experiments
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # fetch dataloaders
    _, val_loader = dataloader.get_bev_dataloaders(cfg)

    # Define the model
    model = get_network(params).to(device)

    # fetch loss function and metrics
    loss_fn = get_loss_fn(params)
    # num_classes+1 for background.
    metrics = {}
    for metric in params.metrics:
        metrics[metric] = get_metrics(metric, params)

    # Reload weights from the saved file
    model = utils.load_checkpoint(model,
                                  is_best=True,
                                  ckpt_dir=ckpt_dir,
                                  filename=ckpt_filename)[0]

    # Evaluate
    eval_loss, val_metrics = evaluate(model,
                                      loss_fn,
                                      val_loader,
                                      metrics=metrics)

    best_json_path = os.path.join(cfg.logging.log_dir,
                                  f'{cfg.model_name}/{cfg.exp}/evaluation.json')

    for val_metric_name, val_metric_results in val_metrics.items():
        print("{}: {}".format(val_metric_name, val_metric_results))

    utils.save_dict_to_json(val_metrics, best_json_path)

if __name__ == '__main__':
    main()