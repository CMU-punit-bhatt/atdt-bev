"""Defines the neural network"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation import deeplabv3_resnet50


def get_activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['sigmoid', nn.Sigmoid()],
        ['none', nn.Identity()]
    ])[activation]


def get_norm_layer(norm_layer, planes):
    return  nn.ModuleDict([
        ['bn', nn.BatchNorm2d(planes)],
        ['in', nn.InstanceNorm2d(planes)]
    ])[norm_layer]


def conv3x3(in_planes, out_planes):
    """3x3 convolution with padding"""

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, bias=False)

class BasicBlock(nn.Module):

    def __init__(self,
                 inplanes,
                 middleplanes,
                 outplanes,
                 activation='relu',
                 norm_layer=None):

        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, middleplanes)
        self.bn1 = get_norm_layer(norm_layer, middleplanes)
        self.relu = get_activation_func(activation)
        self.conv2 = conv3x3(middleplanes, outplanes)
        self.bn2 = get_norm_layer(norm_layer, outplanes)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class Transfer(nn.Module):
    def __init__(self,
                 inplanes,
                 middleplanes,
                 outplanes,
                 activation='relu',
                 stride=1,
                 dilation=1,
                 norm_layer='bn',
                 num_blocks=3):

        super(Transfer, self).__init__()
        layers = []
        layers.append(BasicBlock(inplanes,
                                 middleplanes,
                                 middleplanes,
                                 activation,
                                 norm_layer))

        for i in range(num_blocks - 2):
            layers.append(BasicBlock(middleplanes,
                                     middleplanes,
                                     middleplanes,
                                     activation,
                                     norm_layer))

        layers.append(BasicBlock(middleplanes,
                                 middleplanes,
                                 outplanes,
                                 activation,
                                 norm_layer))

        self.transfer = nn.Sequential(*layers)

    def forward(self, x):
        x = self.transfer(x)

        return x

class AdaptiveNet(nn.Module):

    def __init__(self, encoder, transfer, decoder):
        super(AdaptiveNet, self).__init__()
        self.encoder = encoder
        self.transfer = transfer
        self.decoder = decoder

    def forward(self, x):

        input_shape = x.shape[-2:]

        x = self.encoder(x)["out"]
        x = self.transfer(x)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        return {"out": x}


def get_network(params):

    if params.arch == 'deeplab_resnet101':
        net = deeplabv3_resnet101(pretrained=params.pretrained)

    if params.arch == 'deeplab_resnet50':
        net = deeplabv3_resnet50(pretrained=params.pretrained)

    net.classifier[-1] = nn.Conv2d(256, params.n_classes, 1, 1)

    return net

def get_transfer(params):

    return Transfer(inplanes=2048, middleplanes=1024, outplanes=2048)

def get_adaptive_network(encoder, transfer, decoder):

    return AdaptiveNet(encoder, transfer, decoder)