import scipy.io as spio

from pose.definitions import SETTINGS, ROOT_PATH
from pose.two.single_person.dataset import SinglePose2DDataset, SinglePose2DDataPoint
from pose.util import Joint2D, Point2DInt, Human2D


def load_mpii_data() -> SinglePose2DDataset:
    base_img_path = ROOT_PATH / SETTINGS["data"]["mpii_base"] / "images"
    lib = spio.loadmat(str(ROOT_PATH / SETTINGS["data"]["mpii_base"] / "annotations"/"mpii_human_pose_v1_u12_1.mat"))
    annolist = lib["RELEASE"][0, 0]["annolist"][0, :]

    data_points = []
    for annot in annolist:
        img_name = annot["image"][0, 0]["name"][0]

        # Skip images with no people
        try:
            annopoints = annot["annorect"][0, 0]["annopoints"]
            joints = []
            mat_joints = annopoints[0, 0]["point"][0, :]
        except (IndexError, KeyError, ValueError, TypeError) as e:
            continue

        for joint in mat_joints:
            x = joint[0][0][0]
            y = joint[1][0][0]
            id = joint[2][0][0]
            is_visible = joint[3].shape == (1, 1)
            joints.append(Joint2D(Point2DInt(x, y), id, is_visible))

        if len(joints) != Human2D.NUM_JOINTS:
            continue
        human = Human2D(joints, (0, 0))
        data_points.append(SinglePose2DDataPoint(str(base_img_path / img_name), human))
    return SinglePose2DDataset(data_points)


if __name__ == "__main__":
    dtst = load_mpii_data()
