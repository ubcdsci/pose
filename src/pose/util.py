import math
from dataclasses import dataclass
from typing import List


@dataclass
class Point2DInt:
    x: int
    y: int

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
        return point.x > self.top_left.x and point.x < self.bottom_right.x and point.y > self.top_left.y and point.y < self.bottom_right.y

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


@dataclass
class Human2D:
    """
    Describes a human as a list of joints and their connectivity as represented on a 2d image.
    """
    joints: List[Point2DInt]
    bbox: BoundingBox

    NUM_JOINTS = 10
