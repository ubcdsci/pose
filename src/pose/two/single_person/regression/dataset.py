from dataclasses import dataclass
from typing import List

import cv2
import torch
import numpy as np

from pose.definitions import TORCH_DEVICE
from pose.util import Human2D


@dataclass
class Pose2DSingleRegressionDataPoint:
    img_path: str
    human: Human2D


@dataclass
class Pose2DSingleRegressionBatch:
    img_tensor: torch.Tensor
    humans_tensor: torch.Tensor


@dataclass
class Pose2DSingleRegressionDataset:
    data: List[Pose2DSingleRegressionDataPoint]

    def create_batch(self, start_idx: int, end_idx: int, stride=1) -> Pose2DSingleRegressionBatch:
        images = []
        humans = []
        for point in self.data[start_idx: end_idx]:
            img = cv2.imread(point.img_path)
            img = cv2.resize(img, (210, 210))[::stride, ::stride, :]
            images.append(np.expand_dims(np.rollaxis(img, 2, 0), 0))
            humans.append(point.human)

        img_tensor = torch.from_numpy(np.concatenate(images, axis=0)).to(TORCH_DEVICE)
        human_tensor = torch.from_numpy(np.concatenate([x.one_hot for x in humans])).to(TORCH_DEVICE)

        return Pose2DSingleRegressionBatch(img_tensor, human_tensor)
