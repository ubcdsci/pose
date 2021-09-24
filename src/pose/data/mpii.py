import cv2
import scipy.io as spio

from pose.definitions import SETTINGS, ROOT_PATH
from pose.two.single_person.regression.dataset import Pose2DSingleRegressionDataset
from pose.util import Joint2D, Point2DInt, Human2D


def _load_mpii_img(img_name: str):
    return cv2.imread(ROOT_PATH / SETTINGS["data"]["mpii_base"] / "images" / img_name)


def load_mpii_data() -> Pose2DSingleRegressionDataset:
    lib = spio.loadmat(ROOT_PATH / SETTINGS["data"]["mpii_base"] / "annotations/mpii_human_pose_v1_u12_1.mat")
    annolist = lib["RELEASE"][0, 0]["annolist"][0, :]
    for annot in annolist:
        img_name = annot["image"][0, 0]["name"][0]
        print(annot["annorect"].dtype)

        # Skip images with no people
        if "annopoints" not in list(annot["annorect"].dtype.fields.keys()):
            continue

        annopoints = annot["annorect"][0, 0]["annopoints"]
        joints = []
        mat_joints = annopoints[0, 0]["point"][0, :]
        for joint in mat_joints:
            x = joint[1][0][0]
            y = joint[2][0][0]
            id = joint[3][0][0]
            is_visible = bool(joint[4][0][0])
            joints.append(Joint2D(Point2DInt(x, y), id, is_visible))
        human = Human2D(joints)


if __name__ == "__main__":
    dtst = load_mpii_data()
