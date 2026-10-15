"""Bind design's object contracts to the unchanged FAFU kitchen and original fingers.

Supported affordances are calibrated explicitly. Attachments and falling are kinematic;
successful commands do not assert grasp forces, thermal state or full collision planning.
"""
import asyncio
import math
import time
from dataclasses import replace
from functools import partial

import numpy as np
from scipy.spatial.transform import Rotation

from wrs_agent.env.kitchen_profile import match_profiles, matrix
from wrs_agent.env.object_motion import Trajectory, tool_pose, transform
from wrs_agent.executor import ExecutionUnknown, SkillFailure
from wrs_agent.skills.robot import SKILLS

OBJECT_SKILLS = ("pick", "reorient", "place_on", "seat_in", "hold_in", "pour",
                 "set_gripper", "reset_scene")
KITCHEN_GUIDE = (
    "FAFU original-mesh kitchen. Use observed object IDs and only advertised skills. "
    "pick verifies closure before lifting; reorient keeps holding. place_on releases; "
    "working/initial placements require calibration. seat_in releases inside a container. "
    "hold_in keeps holding for wall-clock seconds; motion speed never shortens waiting. "
    "pour uses calibrated motion, empties contents and returns upright still held. "
    "Unsupported objects/pairs must fail explicitly. Stop preserves all poses, width and holding. "
    "reset_scene is simulation-only. No full collision, force or thermal verification."
)


class KitchenActions:
    def __init__(self, model, objects):
        self.model, self.specs = model, objects
        self.profiles = match_profiles(objects)
        if not self.profiles:
            raise SkillFailure("kitchen_calibration_missing")
        model.enable_cooking_gripper()
        self.open_width = float(model.gripper.jaw_range[1])
        self.initial = {name: obj.tf.copy() for name, (_, obj) in model.objects.items()}
        self.held, self.attachment = None, None
        self.contents, self.locations = {}, {}
        self.elapsed, self.required, self.wait_complete = 0.0, 10.0, False
        self.interrupted = False
        self.stage, self.skill = "等待动作", ""
        self.frames, self.index = [], 0
        model.manipulation = self

    def obj(self, name):
        if name not in self.model.objects:
            raise SkillFailure("object_not_in_scene")
        return self.model.objects[name][1]

    def profile(self, name):
        self.obj(name)
        if name not in self.profiles:
            raise SkillFailure("object_affordance_not_calibrated")
        return self.profiles[name]

    def require_held(self, name):
        if self.held != name:
            raise SkillFailure("object_not_held")
        return self.profile(name)

    def upright(self, name, tf=None):
        profile = self.profile(name)
        tf = self.obj(name).tf if tf is None else tf
        return float((tf[:3, :3] @ profile.opening_axis)[2]) > .999

    def working_pose(self, name):
        p = self.profile(name)
        if p.upright_rpy is None or p.working_offset is None:
            raise SkillFailure("working_placement_not_calibrated")
        goal = matrix([0., 0., 0.], p.upright_rpy)
        bottom = self.initial[name][:3, 3] + p.working_offset
        bottom[2] = 0.0
        goal[:3, 3] = bottom - goal[:3, :3] @ p.bottom
        return goal

    def container_target(self, name, container, clearance):
        p, vessel = self.profile(name), self.profile(container)
        goal = matrix([0., 0., 0.], p.upright_rpy)
        goal[:3, :3] = Rotation.from_euler("z", 30, degrees=True).as_matrix() @ goal[:3, :3]
        bottom = self.obj(container).tf @ np.r_[vessel.bottom, 1.]
        bottom[2] += clearance
        goal[:3, 3] = bottom[:3] - goal[:3, :3] @ p.bottom
        return goal

    def refresh_attachments(self):
        if self.held:
            self.obj(self.held).tf = self.model.tcp.tf @ self.attachment
        for _ in range(len(self.contents)):
            for child, (parent, offset) in self.contents.items():
                self.obj(child).tf = self.obj(parent).tf @ offset

    def snapshot(self):
        objects = {}
        for name, spec in self.specs.items():
            obj = self.obj(name)
            values = {"pos": obj.pos.tolist(), "rotmat": obj.rotmat.tolist()}
            if name in self.locations:
                values.update(location=self.locations[name], source="simulation",
                              observed_at_ns=time.time_ns())
            objects[name] = spec.model_copy(update=values)
        return {"kinematics": self.model.read(), "objects": objects, "held": self.held,
                "facts": {"manipulation_profile": "fafu-kitchen-v1",
                          "manipulation_stage": self.stage, "last_skill": self.skill,
                          "hold_seconds": self.required, "hold_elapsed_seconds": self.elapsed,
                          "timed_hold_complete": self.wait_complete, "physics_enabled": False,
                          "manipulation_interrupted": self.interrupted,
                          "collision_checked": False, "thermal_verified": False,
                          "gripper_actuation_mode": "original_mesh_rotation"}}

    def reset(self):
        self.held, self.attachment, self.contents, self.locations = None, None, {}, {}
        self.elapsed, self.required, self.wait_complete = 0.0, 10.0, False
        self.interrupted = False
        self.stage, self.skill = "场景已复位", "reset_scene"
        self.model.apply(self.model.poses["home"])
        self.model.gripper.set_opening(self.open_width)
        for name, tf in self.initial.items():
            self.obj(name).tf = tf
        return self.snapshot()

    def plan_pick(self, path, args):
        name = args.object
        p = self.profile(name)
        if self.held:
            raise SkillFailure("gripper_already_holding")
        if p.grasp_pos is None:
            raise SkillFailure("object_grasp_not_calibrated")
        if name in self.contents:
            parent = self.contents[name][0]
            if not self.upright(parent):
                raise SkillFailure("container_not_upright")
        grip = path.objects[name] @ matrix(p.grasp_pos, p.grasp_rpy)
        if p.kind == "food":
            approach = grip.copy()
            approach[2, 3] += .06
        else:
            approach = grip.copy()
            approach[:3, 3] -= .04*grip[:3, 0]
        path.move("到抓取位置旁", approach, 2.0, initial=True)
        path.jaw("调整接近间距", p.approach_opening)
        path.move("接近物体抓取点", grip, 1.2)
        path.jaw("闭爪并检查接触", p.opening, {"kind": "grasp", "name": name})
        path.grasp(name)
        lift = path.tcp_tf.copy()
        lift[2, 3] += .07 if p.kind == "food" else .045 if self.upright(name) else .16
        path.move("提起已夹物体", lift, 1.5)

    def plan_reorient(self, path, args):
        p = self.require_held(args.object)
        if p.upright_rpy is None:
            raise SkillFailure("orientation_not_calibrated")
        goal = self.working_pose(args.object)
        goal[2, 3] += .08
        if args.orientation == "initial":
            goal = self.initial[args.object].copy()
            goal[2, 3] += .16
        elif args.orientation == "upside_down":
            raise SkillFailure("orientation_not_calibrated")
        elif self.upright(args.object):
            path.hold("物体已正放", .04)
            return
        path.move("调整持有物体朝向", goal @ np.linalg.inv(path.attachment), 2.0)

    def release(self, path, name, target):
        p = self.profile(name)
        path.jaw("松开物体", p.opening+.002,
                 {"kind": "release", "name": name, "target": target})
        path.release()
        path.jaw("张开夹爪", self.open_width)

    def plan_place(self, path, args):
        p = self.require_held(args.object)
        surface = self.specs.get(args.surface)
        if surface is None:
            raise SkillFailure("surface_not_in_scene")
        if p.kind != "strainer" or surface.geometry.kind != "box":
            raise SkillFailure("surface_placement_not_calibrated")
        top = surface.pos[2] + surface.geometry.xyz_lengths[2]/2
        if not np.allclose(surface.rotmat, np.eye(3), atol=1e-5):
            raise SkillFailure("surface_not_level")
        if args.placement == "initial":
            goal = self.initial[args.object].copy()
            staging = self.working_pose(args.object)
            staging[2, 3] += .08
            path.move("返回安全工作位", staging @ np.linalg.inv(path.attachment), 2.0)
            above = goal.copy()
            above[2, 3] += .16
            path.move("恢复原放置朝向", above @ np.linalg.inv(path.attachment), 2.8)
        else:
            if not self.upright(args.object):
                raise SkillFailure("held_object_not_upright")
            goal = self.working_pose(args.object)
            if args.placement == "center":
                point = np.array([*surface.pos[:2], top])
                goal[:3, 3] = point - goal[:3, :3] @ p.bottom
        bottom = goal[:3, 3] if args.placement == "initial" else (
            goal @ np.r_[p.bottom, 1.])[:3]
        half = np.asarray(surface.geometry.xyz_lengths[:2])/2
        if (np.abs(bottom[:2]-surface.pos[:2])+p.radius > half+.001).any():
            raise SkillFailure("surface_too_small")
        if abs(bottom[2]-top) > .003:
            raise SkillFailure("placement_surface_height_mismatch")
        path.move("放下物体", goal @ np.linalg.inv(path.attachment), 1.2)
        self.release(path, args.object, args.surface)
        retreat = path.tcp_tf.copy()
        retreat[2, 3] += .09
        path.move("夹爪撤离", retreat, 1.0)

    def plan_insert(self, path, args, *, keep=False):
        p = self.require_held(args.object)
        vessel = self.profile(args.container)
        if not self.upright(args.container):
            raise SkillFailure("container_not_upright")
        if p.radius > vessel.radius:
            raise SkillFailure("object_exceeds_container_opening")
        if any(parent == args.container for parent, _ in self.contents.values()):
            raise SkillFailure("container_occupied")
        if p.kind == "food" and vessel.kind == "strainer" and not keep:
            if p.half_height*2 > np.linalg.norm(np.asarray(vessel.rim)-vessel.bottom):
                raise SkillFailure("container_too_shallow")
            pick = path.objects[args.object][:3, 3]
            path.move("调整搬运姿态", tool_pose(self.model, pick+[0, 0, .26], math.pi/4), 1.8)
            bottom = (path.objects[args.container] @ np.r_[vessel.bottom, 1.])[:3]
            release = bottom+[0, 0, .23]
            path.move("移到容器开口上方", tool_pose(self.model, release, math.pi/4), 3.0)
            self.release(path, args.object, "air")
            target = transform(bottom+[0, 0, p.half_height+.010])
            path.fall(args.object, "物体进入容器", target,
                      {"kind": "contain", "child": args.object, "parent": args.container})
            path.contents[args.object] = (
                args.container, np.linalg.inv(path.objects[args.container]) @ target)
            path.move("夹爪撤离容器", tool_pose(self.model, release+[0, 0, .05], math.pi/4), 1.0)
        elif p.kind == "strainer" and vessel.kind == "vessel":
            if not self.upright(args.object):
                raise SkillFailure("held_object_not_upright")
            above = self.container_target(args.object, args.container, .125)
            path.move("持物移到容器上方", above @ np.linalg.inv(path.attachment), 3.0)
            immersed = self.container_target(args.object, args.container, .010)
            path.move("持物进入容器", immersed @ np.linalg.inv(path.attachment), 1.8)
            if keep:
                path.hold("确认进入容器", .15, {"kind": "wait_begin", "seconds": args.seconds})
                path.hold("计时保持", args.seconds, {"kind": "wait_end"}, timed=True)
            else:
                self.release(path, args.object, args.container)
                path.contents[args.object] = (
                    args.container, np.linalg.inv(path.objects[args.container]) @ immersed)
                path.hold("确认容器归属", .04,
                          {"kind": "contain", "child": args.object, "parent": args.container})
                retreat = path.tcp_tf.copy()
                retreat[2, 3] += .09
                path.move("夹爪撤离容器", retreat, 1.0)
        else:
            raise SkillFailure("insertion_pair_not_calibrated")

    def plan_pour(self, path, args):
        p = self.require_held(args.source)
        target_profile = self.profile(args.target)
        children = [child for child, (parent, _) in self.contents.items() if parent == args.source]
        if p.kind != "strainer" or target_profile.kind != "surface" or not children:
            raise SkillFailure("pour_pair_or_contents_not_calibrated")
        if not self.upright(args.source) or not self.upright(args.target):
            raise SkillFailure("pour_container_not_upright")
        if any(self.profile(child).radius > target_profile.radius for child in children):
            raise SkillFailure("target_surface_too_small")
        source = path.objects[args.source].copy()
        source[2, 3] += .115
        path.move("提起来源容器", source @ np.linalg.inv(path.attachment), 1.8)
        plate = path.objects[args.target][:3, 3]
        axis = path.tcp_tf[:3, 0]
        direction = np.cross(axis, [0, 0, 1])
        basket = plate-.06*direction+[0, 0, .26]
        upright = path.objects[args.source].copy()
        upright[:3, 3] = basket-upright[:3, :3] @ p.bottom
        path.move("移到目标表面上方", upright @ np.linalg.inv(path.attachment), 3.0)
        pivot_local = np.asarray(p.rim)
        pivot_world = (upright @ np.r_[pivot_local, 1.])[:3]
        tipped = upright.copy()
        tipped[:3, :3] = Rotation.from_rotvec(axis*math.radians(105)).as_matrix() @ upright[:3, :3]
        tipped[:3, 3] = pivot_world-tipped[:3, :3] @ pivot_local
        path.move("倾倒来源容器", tipped @ np.linalg.inv(path.attachment), 2.2)
        for child in children:
            path.contents.pop(child)
            lip = transform(pivot_world+.06*(tipped[:3, :3] @ p.opening_axis))
            path.hold("释放容器内容物", .04, {"kind": "detach", "child": child})
            path.fall(child, "内容物滑向开口", lip, None)
            cp = self.profile(child)
            target = path.objects[args.target] @ matrix(
                np.asarray(target_profile.bottom)+[0, 0, cp.half_height])
            path.fall(child, "内容物落到目标表面", target,
                      {"kind": "land", "child": child, "target": args.target})
        path.move("来源容器恢复直立并保持抓持", upright @ np.linalg.inv(path.attachment), 1.5)

    def prepare(self, skill, args, stopped):
        path = Trajectory(self, getattr(args, "motion_speed", 1.0), stopped)
        if skill == "set_gripper":
            if self.held:
                raise SkillFailure("object_held")
            width = (self.open_width if args.command == "open"
                     else float(self.model.gripper.jaw_range[0]))
            path.jaw("调整空夹爪", width, compensate=False)
        else:
            planner = {"pick": self.plan_pick, "reorient": self.plan_reorient,
                       "place_on": self.plan_place, "seat_in": self.plan_insert,
                       "pour": self.plan_pour}.get(skill)
            if skill == "hold_in":
                self.plan_insert(path, args, keep=True)
            else:
                planner(path, args)
        path.check_stop()
        self.frames, self.index = path.frames, 0
        self.goal = path
        return tuple((frame.stage, frame.dt) for frame in path.frames)

    def grasp(self, name):
        p = self.profile(name)
        target = self.obj(name).tf
        center = (target @ np.r_[p.grasp_pos, 1.])[:3]
        if self.held or np.linalg.norm(self.model.tcp.pos-center) > .002:
            raise RuntimeError("grasp_center_mismatch")
        if abs(self.model.width-p.opening) > .0002:
            raise RuntimeError("gripper_not_closed_on_object")
        if p.contact_axis is not None:
            local = (self.model.gripper.contact_points()-target[:3, 3]) @ target[:3, :3]
            if (not np.allclose(np.sort(local[:, p.contact_axis]), p.contact_bounds, atol=.0007)
                    or not ((local[:, 0] >= p.contact_x_bounds[0]-.0007)
                            & (local[:, 0] <= p.contact_x_bounds[1]+.0007)).all()):
                raise RuntimeError("calibrated_contacts_mismatch")
        self.contents.pop(name, None)
        self.held, self.attachment = name, np.linalg.inv(self.model.tcp.tf) @ target
        self.locations[name] = "gripper"

    def tick(self):
        frame = self.frames[self.index]
        self.stage = frame.stage
        self.model.sync(frame.q, frame.width)
        if frame.object_tf is not None:
            self.obj(frame.object_id).tf = frame.object_tf
        event = frame.event or {}
        kind = event.get("kind")
        if kind == "grasp":
            self.grasp(event["name"])
        elif kind == "release":
            name = event["name"]
            if self.held != name:
                raise RuntimeError("release_held_object_mismatch")
            self.held, self.attachment = None, None
            self.locations[name] = event["target"]
        elif kind == "contain":
            child, parent = event["child"], event["parent"]
            self.contents[child] = (parent, np.linalg.inv(self.obj(parent).tf) @ self.obj(child).tf)
            self.locations[child] = parent
        elif kind == "detach":
            self.contents.pop(event["child"], None)
            self.locations[event["child"]] = "air"
        elif kind == "land":
            self.locations[event["child"]] = event["target"]
        elif kind == "wait_begin":
            self.elapsed, self.required, self.wait_complete = 0.0, event["seconds"], False
        elif kind in {"wait_tick", "wait_end"}:
            self.elapsed += frame.dt
            self.wait_complete = kind == "wait_end"
        self.index += 1
        return self.snapshot()

    def start(self, skill):
        self.skill = skill
        self.interrupted = False

    def stop(self):
        self.interrupted = True
        self.stage = "动作已停止，请复位或明确指定下一步"
        return self.snapshot()

    def verify(self, skill):
        if skill == "reset_scene":
            return (self.held is None and abs(self.model.width-self.open_width) < 1e-6
                    and all(np.allclose(self.obj(name).tf, tf, atol=1e-6)
                            for name, tf in self.initial.items()))
        q = np.asarray(self.model.read()["qs"])
        return bool(self.index == len(self.frames) and self.model.read()["valid"]
                    and (q >= self.model.limits[:, 0]).all()
                    and (q <= self.model.limits[:, 1]).all()
                    and self.held == self.goal.held
                    and abs(self.model.width-self.goal.width) < .0002
                    and all(np.allclose(self.obj(name).tf, tf, atol=.002)
                            for name, tf in self.goal.objects.items())
                    and (skill != "hold_in" or self.wait_complete
                         and self.elapsed >= self.required-1e-6))


def publish(state, snapshot):
    state.kinematics, state.objects, state.held = (
        snapshot["kinematics"], snapshot["objects"], snapshot["held"])
    state.facts.update(snapshot["facts"])
    state.pose = None
    state.version += 1


async def advance(call, controller, skill, state, args, stop, progress):
    if stop.is_set():
        publish(state, await call(controller.stop))
        return False
    if skill == "reset_scene":
        publish(state, await call(controller.reset))
        state.pose = "home"
        return not stop.is_set() and await call(controller.verify, skill)
    try:
        timings = await call(controller.prepare, skill, args, stop.is_set)
    except SkillFailure:
        if stop.is_set():
            publish(state, await call(controller.stop))
            return False
        raise
    if stop.is_set():
        publish(state, await call(controller.stop))
        return False
    await call(controller.start, skill)
    duration = sum(dt for _, dt in timings)
    elapsed, deadline, previous = 0.0, time.monotonic(), None
    try:
        for stage, dt in timings:
            if stage == "计时保持" and previous != stage:
                deadline = time.monotonic()
            previous = stage
            if stop.is_set():
                publish(state, await call(controller.stop))
                return False
            publish(state, await call(controller.tick))
            elapsed += dt
            progress(min(elapsed/duration, .999))
            deadline += dt
            try:
                await asyncio.wait_for(stop.wait(), max(0., deadline-time.monotonic()))
            except TimeoutError:
                pass
        if stop.is_set():
            publish(state, await call(controller.stop))
            return False
        return await call(controller.verify, skill)
    except Exception:
        state.kinematics = {**state.kinematics, "valid": False}
        raise ExecutionUnknown("object_motion_state_unknown") from None


async def install_actions(call, model, state, duration):
    from wrs_agent.env.wrs import advance_virtual

    controller = await call(KitchenActions, model, state.objects)
    publish(state, await call(controller.snapshot))
    state.pose = "home"
    result = [replace(SKILLS[name], instructions=KITCHEN_GUIDE).bind(
              partial(advance_virtual, call, model, duration, name))
              for name in ("observe", "move_named_pose", "move_relative")]
    for name in OBJECT_SKILLS:
        entry = replace(SKILLS[name], instructions=KITCHEN_GUIDE)
        if name == "pour":
            entry = replace(entry, description=entry.description.replace("Taught", "Calibrated"),
                            preconditions=("source_held", "target_visible",
                                           "calibrated_trajectory"))
        entry = replace(entry, timeout=300.0)
        result.append(entry.bind(partial(advance, call, controller, name)))
    return result
