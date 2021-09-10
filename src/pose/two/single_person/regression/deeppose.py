"""
@article{
    1312.4659,
    Author = {Alexander Toshev and Christian Szegedy},
    Title = {DeepPose: Human Pose Estimation via Deep Neural Networks},
    Year = {2013},
    Eprint = {arXiv:1312.4659},
    Doi = {10.1109/CVPR.2014.214},
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd

from pose.util import Human2D, Point2D, BoundingBox
from pose.two.estimator import Estimator2D


class DeepPoseJointRegressor(nn.Module):
    def __init__(self):
        """
        The input to the net is an image of 220 ×220 which via stride of 4 is fed into the network.

        Specifics of this implementation come from both:
        - DeepPose: Human Pose Estimation via Deep Neural Networks
        - Imagenet classification with deep convolutional neural networks
        """
        super().__init__()
        # RGB image, 3 channels. Could potentially do grayscale for fewer channels -> higher speed
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(11, 11))
        # The specifics of the local response normalization are not provided in the paper, but these values appear to
        # work well.
        self.lrn = nn.LocalResponseNorm(size=10)
        # The specifics of the pooling are not provided in the paper, but these values appear to work well
        self.pool = nn.AvgPool2d(4)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=(3, 3))
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=(3, 3))
        self.fc1 = nn.Linear(4096, 4096)
        # Output vector is comprised of x and y for each join in the human, so we multiple the size by two
        self.fc2 = nn.Linear(4096, 2 * Human2D.NUM_JOINTS)

    def forward(self, x):
        x = self.lrn(self.conv1(x))
        x = self.pool(x)
        x = self.lrn(self.conv2(x))
        x = self.pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DeepPose(Estimator2D):
    def __init__(self):
        pass

    def _initial_regression(self):
        pass

    def _refine(self):
        pass

    def _normalize_joint(self, joint: Point2D, human_bbox: BoundingBox) -> Point2D:
        return Point2D(
            (joint.x - human_bbox.center.x) / human_bbox.w,
            (joint.y - human_bbox.center.y) / human_bbox.h,
        )

    def _denormalize_joint(self, joint: Point2D, human_bbox: BoundingBox) -> Point2D:
        return Point2D(
            joint.x * human_bbox.w + human_bbox.center.x,
            joint.y * human_bbox.h + human_bbox.center.y
        )

    def _find_human_bbox(self) -> BoundingBox:
        pass

    def estimate(self, image) -> Human2D:
        pass
