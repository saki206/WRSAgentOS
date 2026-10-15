"""Kinematic animation of the two original FAFU mesh fingers, not hardware calibration."""

import math

import numpy as np

MAX_ANGLE = math.radians(60)
PIVOT_X = 0.10705
LEFT_PIVOT_Y, RIGHT_PIVOT_Y = -0.0675, 0.0715
TIP_X = 0.17010080790519714
ARM_X, ARM_Y = TIP_X-PIVOT_X, 0.0265
MAX_OPEN = 0.086
MIN_OPEN = RIGHT_PIVOT_Y-LEFT_PIVOT_Y-2*(
    ARM_X*math.sin(MAX_ANGLE)+ARM_Y*math.cos(MAX_ANGLE)
)


def width_for_angle(angle):
    if not math.isfinite(angle) or not 0 <= angle <= MAX_ANGLE:
        raise ValueError("fafu_gripper_angle_out_of_range")
    width = RIGHT_PIVOT_Y-LEFT_PIVOT_Y-2*(ARM_X*math.sin(angle)+ARM_Y*math.cos(angle))
    return min(MAX_OPEN, max(MIN_OPEN, width))


def angle_for_width(width):
    if not math.isfinite(width) or not MIN_OPEN <= width <= MAX_OPEN:
        raise ValueError("fafu_gripper_width_out_of_range")
    angle = math.asin((RIGHT_PIVOT_Y-LEFT_PIVOT_Y-width)/
                      (2*math.hypot(ARM_X, ARM_Y)))-math.atan2(ARM_Y, ARM_X)
    return float(np.clip(angle, 0, MAX_ANGLE))


def contact_tf(width):
    """Contact midpoint relative to the original URDF tool_link origin."""
    angle = angle_for_width(width)
    tf = np.eye(4)
    tf[:3, 3] = [PIVOT_X+ARM_X*math.cos(angle)-ARM_Y*math.sin(angle)-0.165, 0.002, 0]
    return tf


def split_fingers(geometry):
    """Partition existing triangles by connectivity; never cut or scale an asset."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from wrs.geom.geometry import gen_geom_from_raw

    vertices, faces = geometry.vs, geometry.fs
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    graph = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
                       shape=(len(vertices), len(vertices)))
    count, labels = connected_components(graph, directed=False)
    finger_labels = []
    for label in range(count):
        points = vertices[labels == label]
        low, high = points.min(axis=0), points.max(axis=0)
        if (abs(low[0]-0.07905) < 2e-6 and abs(high[0]-TIP_X) < 2e-6
                and (high[1] < 0 or low[1] > 0)):
            finger_labels.append((float(points[:, 1].mean()), label))
    if len(finger_labels) != 2:
        raise ValueError("fafu_original_finger_geometry_not_calibrated")
    left, right = [label for _, label in sorted(finger_labels)]
    face_labels = labels[faces[:, 0]]
    masks = [(face_labels != left) & (face_labels != right),
             face_labels == left, face_labels == right]
    parts = []
    for mask in masks:
        selected = faces[mask]
        used, inverse = np.unique(selected, return_inverse=True)
        parts.append(gen_geom_from_raw(vertices[used], inverse.reshape(-1, 3)))
    return parts


class OriginalGripper:
    def __init__(self, model):
        from wrs.scene.render_model import RenderModel

        self.model = model
        self.link = next(link for link in model.robot.runtime_lnks if link.name == "link6")
        original, = self.link.visuals
        if not np.allclose(original.loc_tf, np.eye(4)):
            raise ValueError("fafu_original_finger_frame_not_calibrated")
        self.original_geometry = original.geom
        self.visuals = [RenderModel(geom=geom, rgb=original.rgb, alpha=original.alpha)
                        for geom in split_fingers(original.geom)]
        self.link.visuals = self.visuals
        self.jaw_range = np.array([MIN_OPEN, MAX_OPEN])
        self.set_opening(MAX_OPEN)

    @property
    def width(self):
        # Read from the rendered finger meshes, not from a requested target.
        left, right = self.local_finger_vertices()
        width = float(right[:, 1].min()-left[:, 1].max())
        for endpoint in (MIN_OPEN, MAX_OPEN):
            if abs(width-endpoint) < 1e-7:
                return endpoint
        return width

    def local_finger_vertices(self):
        return tuple(visual.geom.vs @ visual.rotmat.T + visual.pos
                     for visual in self.visuals[1:])

    def contact_points(self):
        left, right = self.local_finger_vertices()
        points = []
        for vertices, value in ((left, left[:, 1].max()), (right, right[:, 1].min())):
            face = vertices[np.abs(vertices[:, 1]-value) < 2e-7]
            face = face[np.abs(face[:, 0]-face[:, 0].max()) < 2e-7]
            point = face.mean(axis=0)
            # The inner edge extends along Z; its midpoint is the grasp sample.
            point[2] = 0.0
            points.append(point)
        points = np.asarray(points)
        return points @ self.link.rotmat.T + self.link.pos

    def set_opening(self, width):
        angle = angle_for_width(float(width))
        for visual, sign, y in zip(self.visuals[1:], (1, -1),
                                    (LEFT_PIVOT_Y, RIGHT_PIVOT_Y), strict=True):
            a = sign*angle
            rotation = np.array([[math.cos(a), -math.sin(a), 0],
                                 [math.sin(a), math.cos(a), 0], [0, 0, 1]])
            pivot = np.array([PIVOT_X, y, 0])
            visual.set_pos_rotmat(pivot-rotation@pivot, rotation)
        self.model.tcp.set_loc_tf(contact_tf(float(width)))
