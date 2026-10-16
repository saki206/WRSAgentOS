"""Pure FAFU trajectories shared by individually executable object skills."""
import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from wrs_agent.executor import SkillFailure

DT = 0.04


def transform(pos, rotmat=None):
    tf = np.eye(4)
    tf[:3, 3] = pos
    if rotmat is not None:
        tf[:3, :3] = rotmat
    return tf


def interpolate_tf(start, target, u):
    rotation = Slerp([0, 1], Rotation.from_matrix([start[:3, :3], target[:3, :3]]))
    return transform(start[:3, 3] * (1-u) + target[:3, 3] * u, rotation(u).as_matrix())


def tool_pose(model, pos, pitch=0.0):
    yaw = math.atan2(pos[1] - model.base_pos[1], pos[0] - model.base_pos[0])
    rotmat = Rotation.from_euler("z", yaw).as_matrix() @ Rotation.from_euler("y", pitch).as_matrix()
    return transform(pos, rotmat)


@dataclass(frozen=True)
class Frame:
    stage: str
    q: np.ndarray
    width: float
    dt: float
    event: dict | None = None
    object_id: str | None = None
    object_tf: np.ndarray | None = None


class Trajectory:
    def __init__(self, controller, speed, stopped=lambda: False):
        self.controller, self.model = controller, controller.model
        self.speed, self.stopped = speed, stopped
        self.solver = self.model.robot.get_solver(self.model.chain)
        self.root_tf = self.model.robot.runtime_lnks[self.model.chain.base_lidx].tf
        self.q = self.model.robot.qs[self.model.indices].copy()
        self.width = self.model.width
        self.frames = []
        self.objects = {name: obj.tf.copy() for name, (_, obj) in self.model.objects.items()}
        self.held = controller.held
        self.attachment = None if controller.attachment is None else controller.attachment.copy()
        self.contents = {child: (parent, tf.copy())
                         for child, (parent, tf) in controller.contents.items()}
        self.anchor_count = 0

    @property
    def tcp_tf(self):
        return self.solver.fk(self.q, self.root_tf) @ self.model.tool_offset(self.width)

    def check_stop(self):
        if self.stopped():
            raise SkillFailure("object_planning_stopped")

    def solve(self, target, previous):
        self.check_stop()
        low, high = self.model.limits[:, 0], self.model.limits[:, 1]
        seeds = [previous, self.model.poses["home"], [0, 2.2, 1.8, -1, 0, 0]]

        def residual(q):
            tf = self.solver.fk(q, self.root_tf) @ self.model.tool_offset(self.width)
            rotation = Rotation.from_matrix(target[:3, :3] @ tf[:3, :3].T).as_rotvec()
            return np.r_[tf[:3, 3] - target[:3, 3], 0.15 * rotation]

        candidates = []
        for seed in seeds:
            self.check_stop()
            answer = least_squares(
                residual, np.clip(seed, low + 1e-6, high - 1e-6),
                bounds=(low, high), diff_step=1e-4, max_nfev=160,
                ftol=1e-8, xtol=1e-8, gtol=1e-8,
            )
            tf = self.solver.fk(answer.x, self.root_tf) @ self.model.tool_offset(self.width)
            if (np.linalg.norm(tf[:3, 3] - target[:3, 3]) < 0.0005
                    and np.linalg.norm(Rotation.from_matrix(
                        target[:3, :3] @ tf[:3, :3].T).as_rotvec()) < 0.003):
                candidates.append(answer.x)
                if np.max(np.abs(answer.x - previous)) < 0.35:
                    break
        if not candidates:
            raise SkillFailure("object_waypoint_unreachable")
        self.anchor_count += 1
        return min(candidates, key=lambda value: np.linalg.norm(value - previous)).copy()

    def follow(self):
        if self.held:
            self.objects[self.held] = self.tcp_tf @ self.attachment
        for _ in range(len(self.contents)):
            for child, (parent, offset) in self.contents.items():
                self.objects[child] = self.objects[parent] @ offset

    def move(self, stage, target, seconds, *, initial=False):
        start = self.tcp_tf.copy()
        distance = np.linalg.norm(target[:3, 3] - start[:3, 3])
        angle = np.linalg.norm(Rotation.from_matrix(
            target[:3, :3] @ start[:3, :3].T).as_rotvec())
        count = max(1, math.ceil(distance / 0.025), math.ceil(angle / 0.10))
        if initial:
            anchors = [self.q.copy(), self.solve(target, self.q)]
            count = 1
        else:
            anchors = [self.q.copy()]
            for i in range(1, count + 1):
                q = self.solve(interpolate_tf(start, target, i/count), anchors[-1])
                if np.max(np.abs(q - anchors[-1])) > 0.45:
                    raise SkillFailure("object_ik_branch_jump")
                anchors.append(q)
        arc = sum(float(np.max(np.abs(b-a))) for a, b in zip(anchors, anchors[1:], strict=False))
        seconds = max(seconds, 1.5 * arc)
        n = max(2, math.ceil(seconds / DT))
        for i in range(1, n + 1):
            u = i/n
            x = (u*u*(3 - 2*u)) * count
            index = min(int(x), count - 1)
            q = anchors[index] + (anchors[index+1] - anchors[index]) * (x-index)
            self.frames.append(Frame(stage, q, self.width, seconds/n/self.speed))
        self.q = anchors[-1].copy()
        self.follow()

    def hold(self, stage, seconds, event=None, *, timed=False):
        n = max(1, math.ceil(seconds / DT))
        dt = seconds/n if timed else seconds/n/self.speed
        for i in range(n):
            tick = {"kind": "wait_tick"} if timed else None
            self.frames.append(Frame(stage, self.q.copy(), self.width, dt,
                                     event if i == n-1 else tick))

    def jaw(self, stage, width, event=None, *, compensate=True):
        center = self.tcp_tf.copy()
        initial_offset = self.model.tool_offset(self.width)
        widths = np.linspace(self.width, width, 11)[1:]
        for i, value in enumerate(widths):
            self.check_stop()
            self.width = float(value)
            if compensate and not np.allclose(
                    self.model.tool_offset(self.width), initial_offset, atol=1e-12, rtol=0):
                self.q = self.solve(center, self.q)
            self.frames.append(Frame(stage, self.q.copy(), self.width, DT/self.speed,
                                     event if i == len(widths)-1 else None))
        self.width = width

    def grasp(self, name):
        self.contents.pop(name, None)
        self.held = name
        self.attachment = np.linalg.inv(self.tcp_tf) @ self.objects[name]

    def release(self):
        self.follow()
        self.held, self.attachment = None, None

    def fall(self, name, stage, target, event):
        start = self.objects[name].copy()
        for i in range(1, 17):
            tf = interpolate_tf(start, target, (i/16)**2)
            self.frames.append(Frame(stage, self.q.copy(), self.width, DT/self.speed,
                                     event if i == 16 else None, name, tf))
        self.objects[name] = target.copy()
