"""Real Zenoh, Runtime output bindings and explicit offline perception fixtures."""

import asyncio
import json
import os
import runpy

import numpy as np
import pytest
from conftest import eventually
from llm_fixtures import chat_reply

from wrs_agent.errors import AgentError
from wrs_agent.nodes.camera.client import CameraClient
from wrs_agent.nodes.camera.preview import CameraPreview
from wrs_agent.nodes.grasp.data import read_grasps
from wrs_agent.processes import ROOT, LocalStack, python_command
from wrs_agent.runtime import Runtime
from wrs_agent.schemas import Plan, Step, TaskCancelRequest, TaskRequest, new_id

pytestmark = pytest.mark.zenoh


def preparation(target="box", *, region=False, part=False, goal="Lift the cup by its handle"):
    locate = []
    segment = Step(step_id="segment", skill="segment_object", args={"target": target})
    if region:
        locate = [Step(step_id="locate", skill="locate_object", args={"target": target})]
        segment = Step(step_id="segment", skill="segment_part" if part else "segment_region",
                       args={"part": "handle", "goal": goal} if part else {}, depends_on=["locate"],
                       inputs={"location_id": {"step_id": "locate", "field": "location_id"}})
    return Plan(steps=[*locate, segment,
        Step(step_id="grasps", skill="infer_grasps", depends_on=["segment"],
             inputs={"segment_id": {"step_id": "segment", "field": "segment_id"}}),
    ])


async def test_part_model_configuration_failure_does_not_block_ordinary_voice(
    llm_server, monkeypatch, tmp_path,
):
    monkeypatch.setenv("WRS_PART_LLM_ENV_FILE", str(tmp_path / "missing.env"))
    llm_server.response = chat_reply(json.dumps({"kind": "answer", "text": "ordinary reply"}))
    async with LocalStack(duration=.03, live_model=True, voice_part_grasp=True) as stack:
        system = stack.system
        receipt = await system.send_text("把杯子夹起来")
        grasp = await system.planning(receipt.request_id).wait(timeout=10)
        assert grasp.state == "FAILED" and grasp.error.code == "part_env_file_not_found"
        assert llm_server.requests == []

        receipt = await system.send_text("向左移动两厘米")
        ordinary = await system.planning(receipt.request_id).wait(timeout=10)
        assert ordinary.state == "ANSWER" and ordinary.reason == "ordinary reply"
        assert len(llm_server.requests) == 1


async def test_recognized_voice_goal_prepares_grasps_and_stop_cancels_planning(
    llm_server, monkeypatch,
):
    goal = "把左边的杯子夹起来"
    llm_server.response = {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant",
        "content": json.dumps({
            "intent": "grasp",
            "selection": "leftmost",
            "suggestion": {
                "object": "cup", "action": "lift", "avoid_parts": [],
                "preferred_parts": [{
                    "part": "handle", "reason": "For lifting", "condition": "If present",
                }],
            },
        }),
    }}]}
    monkeypatch.setenv("WRS_PART_LLM_BASE_URL", os.environ["LLM_BASE_URL"])
    monkeypatch.setenv("WRS_PART_LLM_PROTOCOL", "openai_chat")
    monkeypatch.setenv("WRS_PART_LLM_MODEL", "gpt-6.2-sol")
    monkeypatch.setenv("WRS_PART_LLM_REASONING_EFFORT", "medium")
    monkeypatch.setenv("WRS_PART_LLM_API_KEY", "offline-voice-fixture")
    monkeypatch.delenv("WRS_PART_LLM_ENV_FILE", raising=False)
    monkeypatch.delenv("WRS_PART_LLM_PROXY", raising=False)
    monkeypatch.delenv("WRS_GRASP_GOAL", raising=False)
    listen = runpy.run_path(str(ROOT / "examples/voice/08_listen_goals.py"))
    async with LocalStack(
        duration=.03, live_model=True, voice_part_grasp=True,
        bindings=ROOT / "examples/voice/wrs_bindings.toml", asr_script=[goal],
    ) as stack:
        system = stack.system
        monkeypatch.setenv("WRS_AGENT_TOKEN", stack.token)
        fixtures = ROOT / "tests/fixtures"
        address = (stack.endpoint, stack.env_id)
        for node, command in (
            ("camera", (fixtures / "camera_node.py", *address)),
            ("yolo", (fixtures / "vision_node.py", *address)),
            ("segmentation-part",
             (fixtures / "grasp_perception_node.py", "segmentation-part", *address)),
            ("graspnet",
             (fixtures / "grasp_perception_node.py", "graspnet", *address, "segmentation-part")),
        ):
            stack._spawn(node, python_command(*command))
            await system.registry.wait_for(node, timeout=10)

        captured = await system.snapshot(node="wrs")
        request_id = await listen["submit_text"](system, f"机器人，{goal}。", captured)
        planned = await system.planning(request_id).wait(timeout=12)
        assert planned.state == "DONE", planned.reason
        result = await planned.task.wait(timeout=12)
        assert result.state == "SUCCEEDED", result.reason
        status = await system.status()
        history = status["action_history"]
        assert [item["skill"] for item in history] == [
            "locate_object", "segment_part", "infer_grasps",
        ]
        outputs = result.step_outputs.values()
        loc = next(value for value in outputs if "location_id" in value)
        seg = next(value for value in outputs if "segment_id" in value)
        grasps = next(value for value in outputs if "candidate_set_id" in value)
        assert loc["observation_id"] == seg["observation_id"] == grasps["observation_id"]
        data, _ = await read_grasps(system.registry.transport("graspnet"),
                                   grasps["candidate_set_id"], planned.task.id)
        assert data.info.segment.segment_id == seg["segment_id"]
        assert data.info.segment.part.name == "handle"
        text, = llm_server.requests[0]["messages"][0]["content"]
        assert text["type"] == "text" and text["text"].endswith(f"User command: {goal}")
        assert len(llm_server.requests) == 1

        whole_object = await system.start(*preparation("cup", region=True).steps)
        result = await whole_object.wait(timeout=12)
        assert result.state == "SUCCEEDED", result.reason
        grasp = result.step_outputs["grasps"]
        data, _ = await read_grasps(system.registry.transport("graspnet"),
                                   grasp["candidate_set_id"], whole_object.id)
        assert data.info.segment.part is None
        assert len(llm_server.requests) == 1

        llm_server.entered.clear()
        llm_server.gate.clear()
        press = await system.listen_begin()
        await system.listen_end(press.press_id)
        transcript = await eventually(lambda: system.listen_result(press.press_id),
                                      lambda result: not result.capturing)
        assert transcript.text == goal and transcript.receipt is None
        receipt = await system.send_text(transcript.text)  # The same handoff as 09.
        await eventually(llm_server.entered.is_set, bool)
        stop = await system.send_text("停止")
        assert stop.accepted
        assert (await system.planning(receipt.request_id).wait(timeout=4)).state == "STALE"
        llm_server.gate.set()
        assert len((await system.status())["action_history"]) == 6


@pytest.mark.parametrize(
    "segmentation", ["sam3", "segmentation", "segmentation-box", "segmentation-part"],
)
async def test_same_frame_pipeline_ambiguity_and_late_cancel(segmentation, monkeypatch):
    part = segmentation == "segmentation-part"
    region = segmentation in {"segmentation-box", "segmentation-part"}
    target = "cup" if region else "box"
    async with LocalStack(duration=.03) as stack:
        monkeypatch.setenv("WRS_AGENT_TOKEN", stack.token)
        stack._spawn("camera", python_command(ROOT / "tests/fixtures/camera_node.py",
                                               stack.endpoint, stack.env_id))
        if region:
            stack._spawn("yolo", python_command(ROOT / "tests/fixtures/vision_node.py",
                                                stack.endpoint, stack.env_id))
            await stack.system.registry.wait_for("yolo", timeout=10)
        for kind in (segmentation, "graspnet"):
            stack._spawn(kind, python_command(ROOT / "tests/fixtures/grasp_perception_node.py",
                                              kind, stack.endpoint, stack.env_id, segmentation))
        registry = stack.system.registry
        for kind in ("camera", segmentation, "graspnet"):
            await registry.wait_for(kind, timeout=10)
        segment_skill = "segment_part" if part else "segment_region" if region else "segment_object"
        skills = {segment_skill: segmentation, "infer_grasps": "graspnet"}
        if region:
            skills["locate_object"] = "yolo"
        runtime = Runtime(stack.system.clients, skills, registry=registry)
        try:
            if region:
                skipped_mask = Plan(steps=[
                    Step(step_id="locate", skill="locate_object", args={"target": target}),
                    Step(step_id="grasps", skill="infer_grasps", depends_on=["locate"],
                         inputs={"segment_id": {"step_id": "locate", "field": "location_id"}}),
                ])
                await runtime.start(TaskRequest(request_id=new_id(), plan=skipped_mask))
                failed = await eventually(runtime.snapshot, lambda s: s["state"] == "FAILED")
                assert failed["error"]["code"] == "artifact_type_mismatch"
                assert not runtime.action_history  # Reject using contracts fetched over Zenoh.
            receipt = await runtime.start(TaskRequest(request_id=new_id(),
                                                       plan=preparation(target, region=region,
                                                                        part=part)))
            result = await eventually(runtime.snapshot,
                                      lambda s: s["state"] in {"SUCCEEDED", "FAILED"}, timeout=8)
            assert result["state"] == "SUCCEEDED", result
            aid = next(a["action_id"] for a in runtime.action_history
                       if a["skill"] == "infer_grasps")
            output = (await stack.system.clients["graspnet"].status(aid)).output
            assert result["step_outputs"]["grasps"] == output
            assert runtime.task_status(receipt["task_id"])["step_outputs"]["grasps"]["count"] == 1
            data, age = await read_grasps(registry.transport("graspnet"),
                                          output["candidate_set_id"], receipt["task_id"])
            assert data.info.segment.observation.observation_id == output["observation_id"]
            assert data.info.owner.task_id == data.info.segment.owner.task_id == receipt["task_id"]
            assert age >= .3 and data.info.segment.observation.source == "synthetic"
            if segmentation == "segmentation":
                assert data.info.segment.score_kind == "clip_cosine"
                assert data.info.segment.text_weights_sha256 == "1" * 64
            if region:
                location = data.info.segment.location
                assert location.observation == data.info.segment.observation
                assert location.owner.task_id == receipt["task_id"]
                assert data.info.segment.score_kind == "box_iou"
            if part:
                assert data.info.segment.part.name == "handle"
                assert data.info.segment.part.grounding_model == "gpt-6.2-sol"
                assert data.info.segment.segment_id == (
                    result["step_outputs"]["segment"]["segment_id"]
                )
                assert data.info.segment.observation.observation_id == (
                    result["step_outputs"]["locate"]["observation_id"]
                )
            latest, _, _ = await CameraClient(registry.transport("camera")).read()
            assert latest.observation_id != output["observation_id"]
            ui = CameraPreview(endpoint=stack.endpoint, env_id=stack.env_id,
                               segmentation_node_id=segmentation)
            await ui.start()

            def preview(view):
                # A browser keeps requesting the view while a new Zenoh session discovers it.
                response = ui.http_response(f"/camera/preview?view={view}", None)
                return json.loads(response.body)

            try:
                grasp_preview = await eventually(
                    lambda: preview("grasps"),
                    lambda value: value.get("candidate_set_id") == output["candidate_set_id"],
                    timeout=4,
                )
                assert grasp_preview["state"] == "ready"
                assert grasp_preview["observation_id"] == output["observation_id"]
                assert len(grasp_preview["grasps"]) == 1
                assert 0 < grasp_preview["segmented_point_count"] < len(
                    grasp_preview["points"],
                )
                mask = await eventually(
                    lambda: preview("segmentation"),
                    lambda value: value.get("segment_id") == data.segment.info.segment_id,
                    timeout=4,
                )
                assert mask["state"] == "ready"
                assert mask["observation_id"] == output["observation_id"]
                assert mask["segment_id"] == data.segment.info.segment_id
                assert mask["segmentation"].startswith("data:image/jpeg;base64,")
                assert mask["pixel_count"] == data.segment.info.pixel_count
                if region:
                    bad_args = {"location_id": location.location_id}
                    if part:
                        bad_args.update(part="handle", goal="Lift the cup by its handle")
                    bad_location = Plan(steps=[Step(step_id="reuse", skill=segment_skill,
                                                    args=bad_args)])
                    await runtime.start(TaskRequest(request_id=new_id(), plan=bad_location))
                    failed = await eventually(runtime.snapshot, lambda s: s["state"] == "FAILED")
                    assert failed["error"]["code"] == "artifact_task_mismatch"
                    unavailable = await eventually(
                        lambda: preview("segmentation"),
                        lambda value: value["reason"] == "segment_not_available",
                        timeout=4,
                    )
                    assert "segmentation" not in unavailable
                replacement = await runtime.start(TaskRequest(
                    request_id=new_id(), plan=preparation(target, region=region, part=part),
                ))
                fresh = await eventually(runtime.snapshot,
                                         lambda s: s["state"] in {"SUCCEEDED", "FAILED"}, timeout=8)
                assert fresh["state"] == "SUCCEEDED", fresh
                current = await eventually(
                    lambda: preview("grasps"),
                    lambda value: value.get("task_id") == replacement["task_id"],
                    timeout=4,
                )
                assert current["state"] == "ready"
                assert current["candidate_set_id"] != output["candidate_set_id"]
                with pytest.raises(AgentError, match="artifact_expired"):
                    await read_grasps(registry.transport("graspnet"),
                                      output["candidate_set_id"], receipt["task_id"])
            finally:
                await ui.close()
            np.testing.assert_allclose(data.segment.arrays["points"][:, :, 2],
                                       data.segment.arrays["depth"] * .001)
            # A new task cannot use another task's segment reference.
            bad_plan = Plan(steps=[Step(step_id="reuse", skill="infer_grasps",
                                        args={"segment_id": data.info.segment.segment_id})])
            await runtime.start(TaskRequest(request_id=new_id(), plan=bad_plan))
            await eventually(runtime.snapshot, lambda s: s["state"] == "FAILED")
            if part:
                await runtime.start(TaskRequest(
                    request_id=new_id(),
                    plan=preparation(target, region=True, part=True, goal="missing handle"),
                ))
                failed = await eventually(runtime.snapshot, lambda s: s["state"] == "FAILED")
                assert failed["error"]["code"] == "part_not_found"
                assert runtime.action_history[-1]["skill"] == "segment_part"
            await runtime.start(TaskRequest(request_id=new_id(),
                                             plan=preparation("ambiguous", region=region,
                                                              part=part)))
            await eventually(runtime.snapshot, lambda s: s["state"] == "FAILED")
            first_skill = "locate_object" if region else "segment_object"
            assert runtime.action_history[-1]["skill"] == first_skill
            cancelled = await runtime.start(TaskRequest(request_id=new_id(),
                                                         plan=preparation(target, region=region,
                                                                          part=part)))
            await eventually(runtime.snapshot, lambda s: bool(s["active_actions"]))
            await runtime.cancel(TaskCancelRequest(request_id=new_id(),
                                                   task_id=cancelled["task_id"]))
            await eventually(runtime.snapshot, lambda s: s["state"] == "CANCELLED")
            await asyncio.sleep(.4)
            assert runtime.action_history[-1]["skill"] == first_skill
            if region:
                cancelled = await runtime.start(TaskRequest(
                    request_id=new_id(), plan=preparation(target, region=True, part=part),
                ))
                await eventually(
                    runtime.snapshot,
                    lambda s: bool(s["active_actions"])
                    and runtime.action_history[-1]["skill"] == segment_skill,
                )
                await runtime.cancel(TaskCancelRequest(request_id=new_id(),
                                                       task_id=cancelled["task_id"]))
                await eventually(runtime.snapshot, lambda s: s["state"] == "CANCELLED")
                await asyncio.sleep(.4)
                assert runtime.action_history[-1]["skill"] == segment_skill
        finally:
            await runtime.close()


async def test_missing_segmentation_cause_survives_zenoh_action_status(monkeypatch):
    async with LocalStack(duration=.03) as stack:
        monkeypatch.setenv("WRS_AGENT_TOKEN", stack.token)
        stack._spawn("graspnet", python_command(
            ROOT / "tests/fixtures/grasp_perception_node.py", "graspnet",
            stack.endpoint, stack.env_id, "segmentation",
        ))
        registry = stack.system.registry
        await registry.wait_for("graspnet", timeout=10)
        runtime = Runtime(stack.system.clients, {"infer_grasps": "graspnet"}, registry=registry)
        try:
            task = await runtime.start(TaskRequest(request_id=new_id(), plan=Plan(steps=[
                Step(step_id="infer", skill="infer_grasps", args={"segment_id": "unused"}),
            ])))
            result = await eventually(runtime.snapshot, lambda s: s["state"] == "FAILED")
            error = result["error"]
            assert error["code"] == "segmentation_unavailable"
            assert error["node_id"] == "segmentation" and error["stage"] == "discovery"
            assert error["task_id"] == task["task_id"] and "FastSAM" in error["message"]
            status = await stack.system.clients["graspnet"].status(error["action_id"])
            assert status.error.model_dump() == error and not status.output
        finally:
            await runtime.close()
