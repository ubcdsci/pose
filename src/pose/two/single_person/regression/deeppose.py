import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd

from pose.two.estimator import Estimator2D
from pose.util import Human2D


class DeepPoseJointRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        # RGB image, 3 channels. Could potentially do grayscale for fewer channels -> higher speed
        n_channels = 3
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels * 2, kernel_size=(12, 12))
        self.pool = nn.MaxPool2d(4)
        self.conv2 = nn.Conv2d(in_channels=n_channels * 2, out_channels=n_channels * 3, kernel_size=(6, 6))
        self.fc1 = nn.Linear(1296, 120)
        self.fc2 = nn.Linear(120, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
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

    def _normalize(self):
        pass




    def estimate(self, image) -> Human2D:
        pass

