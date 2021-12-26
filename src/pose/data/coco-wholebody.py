from typing import Union

from pose.data.data import DATA_PATH
from pose.definitions import SETTINGS
from pose.two.multi_person.dataset import MultiPose2DDataset
from pose.two.single_person.dataset import SinglePose2DDataset
import pandas as pd


def load_val_2014(multi=True) -> Union[SinglePose2DDataset, MultiPose2DDataset]:
    base_data_path = DATA_PATH / SETTINGS["data"]["coco_wholebody_base"]
    images_path = base_data_path / "val2014/images"
    labels_path = base_data_path / "val2014/labels"
    label_df = pd.read_csv(str(labels_path / "foid_labels_bbox_v020.csv"))
    return SinglePose2DDataset(data_points)


if __name__ == "__main__":
    dtst = load_val_2014()
