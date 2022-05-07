import hydra
import torch
from albumentations import Compose, Resize
from omegaconf import DictConfig
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

from utils.models import get_adaptive_network, get_network, get_transfer
from utils.utils import load_checkpoint, RunningAverage
from utils.metrics import get_metrics
from utils.losses import get_loss_fn
from dataloaders.dataloader import get_test_dataloader


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

    # use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() and \
        cfg.training.use_cuda else 'cpu')

    model = get_adaptive_net(cfg).to(device)
    model.eval()

    test_loader = get_test_dataloader(cfg)

    # fetch loss function and metrics
    loss_fn = get_loss_fn(params)
    running_loss = RunningAverage()

    metrics = OrderedDict({})
    for metric in params.metrics:
        metrics[metric] = get_metrics(metric, params)

    for (xb, yb) in tqdm(test_loader):
        xb = xb.to(device)
        yb = yb.to(device)
        output = model(xb)['out']

        if loss_fn is not None:
            loss_b = loss_fn(output, yb)
            running_loss.update(loss_b.item())
        if metrics is not None:
            for _, metric in metrics.items():
                metric.add(output, yb)

        # Need to add some changes for saving or viz here.

    for metric_name, metric in metrics.items():
        print(f'{metric_name}: {metric}')

if __name__ == '__main__':
    main()
