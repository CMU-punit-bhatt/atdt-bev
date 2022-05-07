import torch
import torch.nn as nn

class Masked_L1_loss(nn.Module):
    def __init__(self, threshold=100):
        super(Masked_L1_loss).__init__()

        self.threshold = threshold
        self.e = 1e-10

    def forward(self, prediction, target):

        gt = target.clone()
        prediction = prediction.squeeze(dim=1)
        valid_map = gt>0
        gt[gt>self.threshold] = self.threshold
        gt /= self.threshold
        error = torch.abs(gt[valid_map]-prediction[valid_map])/torch.sum(valid_map)

        return torch.sum(error)

def get_loss_fn(params):

    if params.loss_fn == 'l1':
        return Masked_L1_loss(threshold=params.threshold)

    if params.loss_fn == 'l2':
        return nn.MSELoss()

    return nn.CrossEntropyLoss(ignore_index=params.ignore_index)
