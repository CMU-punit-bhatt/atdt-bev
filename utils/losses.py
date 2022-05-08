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

    weights = torch.ones(params.n_classes)
    weights[[0, 3]] = 0.0075
    weights[[7]] = 0.5
    weights[[4, 10]] = 2
    
    focal_loss = torch.hub.load(
        'adeelh/pytorch-multi-class-focal-loss',
        model='FocalLoss',
        alpha=weights.cuda(),
        gamma=2,
        reduction='sum',
        force_reload=False)
    
    if params.loss_fn == 'l1':
        return Masked_L1_loss(threshold=params.threshold)

    if params.loss_fn == 'l2':
        return nn.MSELoss()

    # return nn.CrossEntropyLoss(weight=weights.cuda(), ignore_index=params.ignore_index)
    return focal_loss

