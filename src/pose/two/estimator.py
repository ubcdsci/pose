from abc import abstractmethod
from typing import List, Union

from pose.util import Human2D


class Estimator2D:
    @abstractmethod
    def estimate(self, image) -> Union[Human2D, List[Human2D]]:
        pass

    def __call__(self, image) -> Union[Human2D, List[Human2D]]:
        return self.estimate(image)
