from abc import abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class PoseBatch:
    img_tensor: torch.Tensor
    humans_tensor: torch.Tensor


class PoseDataset:
    @abstractmethod
    def create_batch(self, start_idx: int, end_idx: int, size=None):
        pass
