"""Parallel translation of the original FAFU fingers, not hardware calibration."""

import math

import numpy as np

TIP_X = 0.17010080790519714
GRASP_DEPTH = 0.008
CONTACT_X = TIP_X-GRASP_DEPTH
MAX_OPEN = 0.086
# Preserve the existing opening limits while changing the simulated motion.
MIN_OPEN = 0.00329279725011425


def closing_travel(width):
    """Each finger moves half the reduction in opening, along local Y only."""
    if not math.isfinite(width) or not MIN_OPEN <= width <= MAX_OPEN:
        raise ValueError("fafu_gripper_width_out_of_range")
    return (MAX_OPEN-width)/2


def contact_tf(width):
    """Symmetric translation keeps the contact midpoint fixed relative to the hand."""
    closing_travel(width)
    tf = np.eye(4)
    tf[:3, 3] = [CONTACT_X-0.165, 0.002, 0]
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
        for visual, vertices, value in zip(
                self.visuals[1:], (left, right),
                (left[:, 1].max(), right[:, 1].min()), strict=True):
            triangles = vertices[visual.geom.fs]
            triangles = triangles[np.all(np.abs(triangles[:, :, 1]-value) < 2e-7, axis=1)]
            # Verify the deeper sample is on an actual inner-pad triangle, not
            # merely within the finger's bounding box or at a decorative rib.
            origin = triangles[:, 0, [0, 2]]
            edge_b = triangles[:, 1, [0, 2]]-origin
            edge_c = triangles[:, 2, [0, 2]]-origin
            det = edge_b[:, 0]*edge_c[:, 1]-edge_b[:, 1]*edge_c[:, 0]
            delta = np.array([CONTACT_X, 0.])-origin
            usable = np.abs(det) > 1e-12
            u = np.zeros(len(det))
            v = np.zeros(len(det))
            u[usable] = (delta[:, 0]*edge_c[:, 1]-delta[:, 1]*edge_c[:, 0])[usable]/det[usable]
            v[usable] = (edge_b[:, 0]*delta[:, 1]-edge_b[:, 1]*delta[:, 0])[usable]/det[usable]
            if not np.any(usable & (u >= -1e-6) & (v >= -1e-6) & (u+v <= 1+1e-6)):
                raise ValueError("fafu_inner_contact_pad_not_calibrated")
            points.append([CONTACT_X, value, 0.])
        points = np.asarray(points)
        return points @ self.link.rotmat.T + self.link.pos

    def set_opening(self, width):
        travel = closing_travel(float(width))
        for visual, sign in zip(self.visuals[1:], (1, -1), strict=True):
            visual.set_pos_rotmat(np.array([0., sign*travel, 0.]), np.eye(3))
        self.model.tcp.set_loc_tf(contact_tf(float(width)))
