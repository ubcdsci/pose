from dataclasses import dataclass
from typing import List

import numpy as np

from pose.util import Human2D


@dataclass
class Pose2DSingleRegressionDataPoint:
    img: np.ndarray
    human: Human2D



@dataclass
class Pose2DSingleRegressionDataset:
    def __init__(self):
        pass
