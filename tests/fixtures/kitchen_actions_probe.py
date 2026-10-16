"""Real original-mesh geometry and independent object contracts, without a renderer."""
import json

import numpy as np

from wrs_agent import step
from wrs_agent.env.fafu_backend import FafuModel
from wrs_agent.env.kitchen_actions import KitchenActions
from wrs_agent.env.wrs import ROOT, sync_scene_objects
from wrs_agent.recipes.noodles import noodle_steps
from wrs_agent.scene import load_scene
from wrs_agent.skills.robot import SKILLS


def main():
    model = FafuModel()
    original = load_scene(ROOT / "wrs_agent/assets/kitchen/scene.toml")
    # Exercise a second ID vocabulary, proving the handler uses observed IDs and asset profiles.
    specs = model.load_scene({"observed_"+name: obj for name, obj in original.items()})
    controller = KitchenActions(model, specs)
    assert len(specs) == 8 and abs(model.width-.086) < 1e-7
    gripper = model.gripper
    assert sum(len(v.geom.fs) for v in gripper.visuals) == len(gripper.original_geometry.fs)
    assert all(np.allclose(v.loc_tf, np.eye(4)) for v in gripper.visuals)
    stationary = gripper.visuals[0].geom.vs.copy()
    original_fingers = tuple(points.copy() for points in gripper.local_finger_vertices())
    original_contact = model.tcp.tf.copy()
    original_q = model.read()["qs"]
    for width in (.060, .040, .02447608876694598, gripper.jaw_range[0], .086):
        gripper.set_opening(float(width))
        assert abs(gripper.width-width) < 2e-7
        for visual, original_points, points in zip(
                gripper.visuals[1:], original_fingers, gripper.local_finger_vertices(),
                strict=True):
            assert np.array_equal(visual.rotmat, np.eye(3))
            delta = points-original_points
            assert np.allclose(delta[:, [0, 2]], 0., atol=1e-12)
            assert np.allclose(delta[:, 1], delta[0, 1], atol=1e-12)
        assert np.array_equal(model.tcp.tf, original_contact)
        assert model.read()["qs"] == original_q
    assert controller.snapshot()["facts"]["gripper_actuation_mode"] == (
        "original_mesh_parallel_translation")
    viewer_scene, displayed = model.wrs.wss.Scene(), {}
    sync_scene_objects(model.wrs, viewer_scene, controller.snapshot()["objects"], displayed)
    rendered_bodies = {name: body for name, (_, body) in displayed.items()}
    wrist = gripper.link
    grabs = []
    planned_seconds = 0.0
    for item in noodle_steps(specs, seconds=10.0, motion_speed=3.0):
        args = SKILLS[item.skill].arguments.model_validate(item.args)
        before = controller.snapshot()
        timings = controller.prepare(item.skill, args, lambda: False)
        planned_seconds += sum(dt for _, dt in timings)
        after_plan = controller.snapshot()
        assert after_plan["kinematics"]["qs"] == before["kinematics"]["qs"]
        assert after_plan["held"] == before["held"]
        assert after_plan["kinematics"]["gripper_width"] == before["kinematics"]["gripper_width"]
        for name, obj in before["objects"].items():
            assert obj.model_dump(exclude={"observed_at_ns"}) == (
                after_plan["objects"][name].model_dump(exclude={"observed_at_ns"}))
        controller.start(item.skill)
        source_pick = item.skill == "pick" and not grabs
        for frame in controller.frames:
            before_jaw_q = model.read()["qs"]
            before_jaw_tcp = model.tcp.tf.copy()
            controller.tick()
            if frame.stage in {"调整接近间距", "闭爪并检查接触", "松开物体", "张开夹爪"}:
                assert model.read()["qs"] == before_jaw_q
                assert np.allclose(model.tcp.tf, before_jaw_tcp, atol=1e-12)
                assert all(np.array_equal(v.rotmat, np.eye(3)) for v in gripper.visuals[1:])
            if frame.event and frame.event["kind"] == "grasp":
                name = frame.event["name"]
                profile = controller.profile(name)
                assert abs(model.width-profile.opening) < 2e-7
                assert np.linalg.norm(model.tcp.pos - (
                    controller.obj(name).tf @ np.r_[profile.grasp_pos, 1.])[:3]) < .0005
                grabs.append(profile.kind)
                # Both samples must be inside the original pads, 8 mm behind the tips.
                local_contacts = (gripper.contact_points()-gripper.link.pos) @ gripper.link.rotmat
                assert np.allclose(local_contacts[:, 0], .17010080790519714-.008, atol=1e-7)
            if frame.event or controller.index % 5 == 0:
                sync_scene_objects(model.wrs, viewer_scene, controller.snapshot()["objects"],
                                   displayed)
                for name, (_, rendered) in displayed.items():
                    assert rendered is rendered_bodies[name]
                    assert np.allclose(rendered.tf, controller.obj(name).tf, atol=1e-7)
            if source_pick:
                if frame.stage in {"接近物体抓取点", "闭爪并检查接触", "提起已夹物体"}:
                    for visual in gripper.visuals:
                        local = visual.geom.vs @ visual.rotmat.T+visual.pos
                        world = local @ wrist.rotmat.T+wrist.pos
                        assert world[:, 2].min() >= .002, (frame.stage, float(world[:, 2].min()))
        assert controller.verify(item.skill), item.skill
        if item.skill == "seat_in":
            assert controller.held is None
        if item.skill == "hold_in":
            assert abs(sum(dt for stage, dt in timings if stage == "计时保持")-10.) < 1e-8
            assert controller.held == "observed_skimmer"
        if item.skill == "pour":
            assert controller.held == "observed_skimmer"
            assert controller.locations["observed_plate_block"] == "observed_plate"
        assert np.array_equal(stationary, gripper.visuals[0].geom.vs)
    assert grabs == ["strainer", "food", "strainer"]
    assert controller.held is None and abs(model.width-.086) < 1e-7
    for name in ("observed_skimmer", "observed_plate_block"):
        assert np.allclose(controller.obj(name).tf, controller.initial[name], atol=.002)
    controller.reset()
    assert controller.verify("reset_scene")
    assert (planned_seconds-10.0)*12+120 < 600  # Slowest legal speed/wait recipe fits the budget.

    # seat_in releases the strainer; it remains a different contract from hold_in.
    alternate = noodle_steps(specs, group="load", motion_speed=3.0) + [
        step("pick", object="observed_skimmer", motion_speed=3.0),
        step("seat_in", object="observed_skimmer", container="observed_pot", motion_speed=3.0),
    ]
    for item in alternate:
        args = SKILLS[item.skill].arguments.model_validate(item.args)
        controller.prepare(item.skill, args, lambda: False)
        for _ in controller.frames:
            controller.tick()
        assert controller.verify(item.skill), item.skill
    assert controller.held is None
    assert controller.locations["observed_skimmer"] == "observed_pot"
    assert controller.locations["observed_plate_block"] == "observed_skimmer"
    assert "observed_skimmer" in controller.contents
    controller.reset()
    for command in ("close", "open"):
        q = model.read()["qs"]
        args = SKILLS["set_gripper"].arguments(command=command)
        controller.prepare("set_gripper", args, lambda: False)
        for _ in controller.frames:
            controller.tick()
        assert controller.verify("set_gripper")
        assert model.read()["qs"] == q
        assert controller.held is None
    print(json.dumps({"result": "PASS", "renamed_ids": True, "nine_skills": True,
                      "original_geometry": True, "contacts": True, "ten_seconds_planned": True,
                      "seat_in_releases": True, "raw_jaw_keeps_arm_fixed": True,
                      "parallel_translation": True, "stable_render_objects": True,
                      "grasp_depth_mm": 8}),
          flush=True)


if __name__ == "__main__":
    main()
