import albumentations
import cv2
import matplotlib.pyplot as plt
import numpy as np


class IPMConfig:
    """
    Stores the coordinates for key pixels in the front and the corresponding pixels in the BEV.
    """
    front_A = [216, 260]
    front_B = [270, 260]
    front_C = [512, 512]
    front_D = [0, 512]

    bev_A = [140, 0]
    bev_B = [300, 0]
    bev_C = [284, 512]
    bev_D = [154, 506]

    # TODO: Add a nicer way to update the coordinates

    @staticmethod
    def get_input_pts():
        return np.float32([IPMConfig.front_A, IPMConfig.front_B, IPMConfig.front_C, IPMConfig.front_D])

    @staticmethod
    def get_output_pts():
        return np.float32([IPMConfig.bev_A, IPMConfig.bev_B, IPMConfig.bev_C, IPMConfig.bev_D])


def apply_transform(image, **params):
    # Example M tranformation matrix.
    # M = np.array([[-2.91978535e-01, -9.15210483e-01,  2.86843213e+02],
    #               [ 0.00000000e+00, -2.36968524e+00,  6.16118162e+02],
    #               [ 2.70117061e-05, -4.25812392e-03,  1.00000000e+00]])

    M = cv2.getPerspectiveTransform(IPMConfig.get_input_pts(), IPMConfig.get_output_pts())
    return cv2.warpPerspective(image, M, (512, 512),flags=cv2.INTER_NEAREST)


def create_lambda_transform():
    return albumentations.augmentations.transforms.Lambda(image=apply_transform)
