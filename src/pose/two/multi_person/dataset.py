from dataclasses import dataclass
from typing import List

import cv2
import torch
import numpy as np

from pose.dataset import PoseBatch, PoseDataset
from pose.definitions import TORCH_DEVICE
from pose.util import Human2D


@dataclass
class MultiPose2DDataPoint:
    img_path: str
    human: Human2D

@dataclass
class MultiPose2DDataset(PoseDataset):
    data: List[MultiPose2DDataPoint]

    def create_batch(self, start_idx: int, end_idx: int, size=None) -> PoseBatch:
        images = []
        human_arrs = []
        for point in self.data[start_idx: end_idx]:
            img = cv2.imread(point.img_path)
            img_shape = img.shape[:2]
            if size is not None:
                img = cv2.resize(img, size)

            # point.human.resize(img_shape, size)
            images.append(np.expand_dims(np.rollaxis(img, 2, 0), 0))
            human_arrs.append(np.expand_dims(point.human.one_hot(img_shape), 0))

        img_tensor = torch.from_numpy(np.concatenate(images, axis=0)).to(TORCH_DEVICE) / 255.0
        human_tensor = torch.from_numpy(np.concatenate(human_arrs)).to(TORCH_DEVICE).float()

        return PoseBatch(img_tensor, human_tensor)
