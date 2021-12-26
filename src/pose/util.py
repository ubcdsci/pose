import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Point2DInt:
    x: int
    y: int


    def __init__(self, x, y):
        self.x = x
        self.y = y
        if self.x > 2000 or self.y > 2000:
            print("asd")

    def sqr_dist_to(self, other):
        return (other.x - self.x) ** 2 + (other.y - self.y) ** 2

    def dist_to(self, other):
        return math.sqrt(self.sqr_dist_to(other))

    def __add__(self, other):
        return Point2DInt(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point2DInt(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Point2DInt(self.x * scalar, self.y * scalar)

    def __floordiv__(self, scalar):
        return Point2DInt(self.x // scalar, self.y // scalar)

    def __truediv__(self, scalar):
        return Point2DInt(self.x // scalar, self.y // scalar)

    def __iter__(self):
        yield self.x
        yield self.y

    def __repr__(self):
        return "(x: " + str(self.x) + ", " + "y: " + str(self.y) + ")"


@dataclass
class Point2D:
    x: float
    y: float

    def sqr_dist_to(self, other):
        return (other.x - self.x) ** 2 + (other.y - self.y) ** 2

    def dist_to(self, other):
        return math.sqrt(self.sqr_dist_to(other))

    def __add__(self, other):
        return Point2DInt(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point2DInt(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Point2DInt(self.x * scalar, self.y * scalar)

    def __floordiv__(self, scalar):
        return Point2DInt(self.x // scalar, self.y // scalar)

    def __truediv__(self, scalar):
        return Point2DInt(self.x / scalar, self.y / scalar)

    def __iter__(self):
        yield self.x
        yield self.y

    def __repr__(self):
        return "(x: " + str(self.x) + ", " + "y: " + str(self.y) + ")"


@dataclass
class BoundingBox:
    # Top left corner, width, and height.
    x: int
    y: int
    w: int
    h: int

    def __iter__(self):
        yield self.x
        yield self.y
        yield self.w
        yield self.h

    """from top left and bottom right corners"""

    @classmethod
    def from_corners(cls, ax, ay, bx, by):
        if ax > bx:
            ax, bx = bx, ax
        if ay > by:
            ay, by = by, ay
        return cls(ax, ay, bx - ax, by - ay)

    @classmethod
    def from_points(cls, pa: Point2DInt, pb: Point2DInt):
        return cls.from_corners(pa.x, pa.y, pb.x, pb.y)

    @classmethod
    def from_center(cls, center: Point2DInt, w, h):
        return cls(*(center - Point2DInt(w, h) / 2), w, h)

    @property
    def top_left(self) -> Point2DInt:
        return Point2DInt(self.x, self.y)

    @property
    def _size_as_point(self) -> Point2DInt:
        return Point2DInt(self.w, self.h)

    @property
    def bottom_right(self) -> Point2DInt:
        return self.top_left + self._size_as_point

    @property
    def center(self) -> Point2DInt:
        return self.top_left + self._size_as_point / 2

    def contains_point(self, point: Point2DInt) -> bool:
        return self.top_left.x < point.x < self.bottom_right.x and self.top_left.y < point.y < self.bottom_right.y

    def contains_box(self, other_box) -> bool:
        return self.contains_point(other_box.top_left) and self.contains_point(other_box.bottom_right)

    def _contains_either_corner(self, other_box) -> bool:
        return self.contains_point(other_box.top_left) or self.contains_point(other_box.bottom_right)

    def intersects(self, other_box) -> bool:
        return self._contains_either_corner(other_box) or other_box._contains_either_corner(self)

    def squarify(self):
        s = max([self.w, self.h])
        return BoundingBox.from_center(self.center, s, s)

    def clamp(self, max_box):
        out_ax = max(self.top_left.x, max_box.x)
        out_ay = max(self.top_left.y, max_box.y)
        out_bx = min(self.bottom_right.x, max_box.x)
        out_by = min(self.bottom_right.y, max_box.y)
        return BoundingBox.from_corners(out_ax, out_ay, out_bx, out_by)


def normalize_human_coords(x, y, human_bbox: BoundingBox) -> Point2D:
    return Point2D(
        (x - human_bbox.center.x) / human_bbox.w,
        (y - human_bbox.center.y) / human_bbox.h,
    )


def denormalize_human_coords(norm_x, norm_y, human_bbox: BoundingBox) -> Point2D:
    return Point2D(
        norm_x * human_bbox.w + human_bbox.center.x,
        norm_y * human_bbox.h + human_bbox.center.y
    )


@dataclass
class Joint2D:
    pos: Point2DInt
    id: int
    is_visible: bool


@dataclass
class   Human2D:
    """
    Describes a human as a list of joints and their connectivity as represented on a 2d image.

    For humans, We use the layout found in the MPII Dataset:

    Joint IDs:
        0 - r ankle,
        1 - r knee,
        2 - r hip,
        3 - l hip,
        4 - l knee,
        5 - l ankle,
        6 - pelvis,
        7 - thorax,
        8 - upper neck,
        9 - head top,
        10 - r wrist,
        11 - r elbow,
        12 - r shoulder,
        13 - l shoulder,
        14 - l elbow,
        15 - l wrist
    """
    NUM_JOINTS = 16

    joints: List[Joint2D]
    img_size: Tuple[int, int]

    @property
    def bbox(self) -> BoundingBox:
        # min_x = min(self.joints, key=lambda x: x.pos.x).pos.x
        # min_y = min(self.joints, key=lambda x: x.pos.y).pos.y
        # max_x = max(self.joints, key=lambda x: x.pos.x).pos.x
        # max_y = max(self.joints, key=lambda x: x.pos.y).pos.y
        return BoundingBox.from_corners(0, 0, self.img_size[0], self.img_size[1])

    def one_hot(self, img_dims) -> np.array:
        out_vec = []
        assert len(self.joints) == Human2D.NUM_JOINTS
        for joint in self.joints:
            img_bbox = BoundingBox.from_corners(0, 0, img_dims[0], img_dims[1])
            norm_joint_pos = normalize_human_coords(joint.pos.x, joint.pos.y, img_bbox)
            out_vec.append(norm_joint_pos.x)
            out_vec.append(norm_joint_pos.y)
        return np.asarray(out_vec)

    def resize(self, current_dims, new_size):
        self.img_size = new_size
        # current dims have channels in dim 0
        x_factor = new_size[0] / current_dims[1]
        y_factor = new_size[1] / current_dims[2]
        for joint in self.joints:
            joint.pos.x *= x_factor
            joint.pos.y *= y_factor
