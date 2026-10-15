import asyncio
import json
import time

import numpy as np
import pytest
from conftest import eventually

from wrs_agent import step
from wrs_agent.processes import NO_WINDOW, ROOT, LocalStack, python_command
from wrs_agent.recipes.noodles import noodle_steps

pytestmark = pytest.mark.wrs
SCENE = ROOT / "wrs_agent/assets/kitchen/scene.toml"
BINDINGS = ROOT / "examples/wrs/bindings.toml"


async def test_original_geometry_and_renamed_ids():
    process = await asyncio.create_subprocess_exec(
        *python_command(ROOT / "tests/fixtures/kitchen_actions_probe.py"), cwd=ROOT,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, creationflags=NO_WINDOW)
    try:
        output, errors = await asyncio.wait_for(process.communicate(), 120)
        assert process.returncode == 0, (output+errors).decode("utf-8")
        assert json.loads(output.decode("utf-8").splitlines()[-1])["result"] == "PASS"
    finally:
        if process.returncode is None:
            process.terminate()
            await process.wait()


@pytest.mark.zenoh
async def test_runtime_composes_atomic_skills_waits_and_cancels_without_pouring():
    async with LocalStack(backend="wrs", scene=SCENE, bindings=BINDINGS) as stack:
        system = stack.system
        features = await system.clients["wrs"].features()
        assert {"pick", "reorient", "place_on", "seat_in", "hold_in", "pour", "reset_scene"} <= (
            set(features.skills))
        assert "cook_noodles" not in features.skills
        initial = await system.snapshot(node="wrs")
        initial_objects = initial.data.objects
        failure = await (await system.action("seat_in", object="plate_block",
                                            container="skimmer")).wait()
        assert failure.state == "FAILED" and failure.reason == "object_not_held"
        after = await system.snapshot(node="wrs")
        assert after.data.robot.kinematics.qs == initial.data.robot.kinematics.qs
        started = time.monotonic()
        task = await system.start(*noodle_steps(initial_objects, motion_speed=3.0))
        result = await task.wait(timeout=180)
        assert result.state == "SUCCEEDED", result
        assert len(result.steps) == 9 and set(result.steps.values()) == {"SUCCEEDED"}
        assert time.monotonic()-started >= 10.0
        served = await system.snapshot(node="wrs")
        assert served.data.robot.held_object is None
        assert served.data.robot.kinematics.gripper_width == pytest.approx(.086)
        assert served.data.objects["plate_block"].location == "plate"
        assert served.data.facts["hold_elapsed_seconds"] >= 9.95
        assert served.data.facts["timed_hold_complete"]
        for name in ("skimmer", "plate_block"):
            assert served.data.objects[name].pos == pytest.approx(
                initial_objects[name].pos, abs=.002)
            assert np.allclose(served.data.objects[name].rotmat,
                               initial_objects[name].rotmat, atol=.002)

        assert (await (await system.action("reset_scene")).wait()).state == "SUCCEEDED"
        closing_task = await system.start(step("pick", object="skimmer", motion_speed=1.0))
        await eventually(lambda: system.snapshot(node="wrs"), lambda snapshot: (
            snapshot.data.facts.get("manipulation_stage") == "闭爪并检查接触"
            and snapshot.data.robot.held_object is None), timeout=30)
        assert (await closing_task.cancel()).accepted
        assert (await closing_task.wait()).state == "CANCELLED"
        stopped = await eventually(lambda: system.snapshot(node="wrs"), lambda s: s.stop_confirmed)
        await asyncio.sleep(.15)
        frozen = await system.snapshot(node="wrs")
        assert frozen.data.robot.kinematics == stopped.data.robot.kinematics
        assert frozen.data.objects == stopped.data.objects
        assert frozen.data.robot.held_object is None
        assert .0244 < frozen.data.robot.kinematics.gripper_width < .04
        if frozen.admission == "HELD":
            assert (await system.allow_actions(node="wrs")).accepted
        assert (await (await system.action("reset_scene")).wait()).state == "SUCCEEDED"

        task = await system.start(*noodle_steps(initial_objects, motion_speed=3.0))
        await eventually(lambda: system.snapshot(node="wrs"), lambda s: (
            s.data.facts.get("manipulation_stage") == "计时保持"
            and s.data.facts.get("hold_elapsed_seconds", 0) >= .2), timeout=60)
        assert (await task.cancel()).accepted
        cancelled = await task.wait()
        assert cancelled.state == "CANCELLED"
        stopped = await eventually(lambda: system.snapshot(node="wrs"), lambda s: s.stop_confirmed)
        assert stopped.data.robot.held_object == "skimmer"
        assert stopped.data.objects["plate_block"].location == "skimmer"
        assert not stopped.data.facts["timed_hold_complete"]
        await asyncio.sleep(.15)
        frozen = await system.snapshot(node="wrs")
        assert frozen.data.robot.kinematics == stopped.data.robot.kinematics
        assert frozen.data.objects == stopped.data.objects
        if frozen.admission == "HELD":
            assert (await system.allow_actions(node="wrs")).accepted
        assert (await (await system.action("reset_scene")).wait()).state == "SUCCEEDED"
