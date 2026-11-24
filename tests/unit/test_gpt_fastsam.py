import asyncio
import hashlib
import json
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from scripts import check_gpt_fastsam as cli
from wrs_agent.errors import AgentError
from wrs_agent.nodes.segmentation.fastsam import FastSamSegmenter
from wrs_agent.nodes.segmentation.node import SegmentationNode
from wrs_agent.nodes.vision.artifacts import ArtifactCache
from wrs_agent.planner import PlanDecision, PlanRequest
from wrs_agent.planner.grasp_part_prompts import PartBox, VoiceGraspSuggestion
from wrs_agent.planner.grasp_parts import VoiceGraspPlanner
from wrs_agent.planner.providers import part_grounding
from wrs_agent.planner.providers.llm import LLMConfig
from wrs_agent.planner.providers.part_grounding import PartGroundingClient, reply_text
from wrs_agent.planner.providers.wire import LLMError


async def test_voice_routes_ordinary_goals_and_clarifies_incomplete_grasps():
    other = PlanDecision(kind="answer", text="ordinary planner")
    base = SimpleNamespace(plan=AsyncMock(return_value=other), last_usage={"output_tokens": 4})
    part = SimpleNamespace(suggest_voice_goal=AsyncMock(side_effect=LLMError("llm_timeout")))
    factory = Mock(return_value=nullcontext(part))
    planner = VoiceGraspPlanner(factory, base)
    for goal in (
        "向左移动两厘米", "移动到抓取位置", "请把机械臂移动到抓取位置",
        "查看上一轮的抓姿", "打开夹爪", "夹紧夹爪", "lift the robot arm",
        "show the last grasp result",
    ):
        request = PlanRequest(user_goal=goal, world={}, skills=[])
        assert await planner.plan(request) is other
        base.plan.assert_awaited_with(request)
    factory.assert_not_called()
    assert planner.last_usage == {"output_tokens": 4}

    base.plan.reset_mock()
    for goal in (
        "抓杯子", "麻烦抓杯子", "把漏勺夹起来", "把杯子夹一下",
        "计算杯子的抓姿", "pick the cup up",
    ):
        request = request.model_copy(update={"user_goal": goal})
        with pytest.raises(LLMError, match="llm_timeout"):
            await planner.plan(request)
        part.suggest_voice_goal.assert_awaited_with(goal)

    request = request.model_copy(update={"user_goal": "夹起来"})
    part.suggest_voice_goal.side_effect = None
    part.suggest_voice_goal.return_value = VoiceGraspSuggestion(
        intent="clarify", selection="only", suggestion=None,
    )
    assert (await planner.plan(request)).kind == "clarify"
    assert planner.last_usage == {}
    base.plan.assert_not_awaited()


async def test_voice_preserves_explicit_instance_and_rejects_invalid_grasp_setup():
    part = SimpleNamespace(suggest_voice_goal=AsyncMock(
        return_value=VoiceGraspSuggestion.model_validate({
            "intent": "grasp", "selection": "leftmost", "suggestion": {
                "object": "box", "action": "lift", "preferred_parts": [
                    {"part": "side panel", "reason": "flat", "condition": "if rigid"},
                ],
            },
        }),
    ))
    base = SimpleNamespace(plan=AsyncMock())
    planner = VoiceGraspPlanner(lambda: nullcontext(part), base)
    request = PlanRequest(user_goal="把左边的盒子夹起来", world={}, skills=[
        {"name": "locate_object"}, {"name": "segment_region"}, {"name": "infer_grasps"},
    ])
    with pytest.raises(AgentError, match="part_grasp_skills_unavailable"):
        await planner.plan(request)
    request.skills.append({"name": "segment_part"})
    decision = await planner.plan(request)
    assert decision.plan.steps[0].args == {"target": "box", "selection": "leftmost"}
    assert decision.plan.steps[1].args["part"] == "side panel"
    with pytest.raises(ValidationError) as malformed:
        VoiceGraspSuggestion(intent="grasp", selection="only", suggestion=None)
    part.suggest_voice_goal.side_effect = malformed.value
    with pytest.raises(AgentError, match="part_suggestion_invalid"):
        await planner.plan(request)
    base.plan.assert_not_awaited()


@pytest.mark.parametrize(("box", "matches"), [
    ([270, 50, 310, 300], True),   # Long handle with an attachment reaching into the object.
    ([385, 300, 425, 365], False),  # Nearby independent object within the old center margin.
    ([379, 300, 425, 365], False),  # Only a thin overlap with the neighboring object.
    ([245, 300, 290, 360], True),   # A normal part wholly within the detector box.
])
async def test_realtime_part_box_checks_overlap_instead_of_its_center(box, matches):
    width, height = 640, 480
    normalized = [round(value / extent * 999) for value, extent in
                  zip(box, (width, height, width, height), strict=True)]
    node = SegmentationNode.__new__(SegmentationNode)
    node.cache = ArtifactCache()
    node.part_client = SimpleNamespace(
        config=SimpleNamespace(model="fixture"),
        ground=AsyncMock(return_value=PartBox(box=normalized, status="matched",
                                             evidence="Fixture part.")),
    )
    location = SimpleNamespace(
        target="strainer", object=SimpleNamespace(box=[220, 280, 380, 400]),
        observation=SimpleNamespace(width=width, height=height),
    )
    args = SimpleNamespace(goal="Lift the strainer", part="handle")
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    if matches:
        result = await node._ground_part(args, location, rgb, time.monotonic(), asyncio.Event())
        assert result.name == "handle"
        assert result.box == pytest.approx(box, abs=1)
    else:
        with pytest.raises(AgentError, match="part_box_outside_object"):
            await node._ground_part(args, location, rgb, time.monotonic(), asyncio.Event())


@pytest.mark.parametrize("protocol", ["openai_responses", "openai_chat"])
async def test_provider_request_with_image(protocol):
    config = LLMConfig(base_url=part_grounding.DEFAULT_URL, model=part_grounding.DEFAULT_MODEL,
                       protocol=protocol, reasoning_effort="medium")

    def handle(request):
        assert request.headers["Authorization"] == "Bearer fake"
        body = json.loads(request.content)
        assert body["model"] == "gpt-6.2-sol" and body["store"] is False
        message = {"role": "assistant", "content": "{}"}
        if protocol == "openai_chat":
            assert request.url.path == "/v1/chat/completions"
            assert body["reasoning_effort"] == "medium"
            assert body["response_format"]["json_schema"]["strict"] is True
            image = body["messages"][0]["content"][1]["image_url"]["url"]
            reply = {"choices": [{"message": message, "finish_reason": "stop"}]}
        else:
            assert request.url.path == "/v1/responses"
            assert body["reasoning"] == {"effort": "medium"}
            assert body["text"]["format"]["strict"] is True
            image = body["input"][0]["content"][1]["image_url"]
            message.update(type="message", content=[{"type": "output_text", "text": "{}"}])
            reply = {"status": "completed", "output": [message]}
        assert image.startswith("data:image/png;base64,iVBOR")
        return httpx.Response(200, json=reply)

    transport = httpx.MockTransport(handle)
    image = Image.new("RGB", (4, 4))
    async with PartGroundingClient(config, "fake", transport=transport) as client:
        assert await client.complete("Locate", {}, name="box", image=image) == "{}"


async def test_provider_timeout_has_safe_error():
    async def handle(_request):
        await asyncio.sleep(1)
        return httpx.Response(500)

    config = LLMConfig(base_url=part_grounding.DEFAULT_URL, model=part_grounding.DEFAULT_MODEL,
                       protocol="openai_responses", timeout_s=0.01)
    async with PartGroundingClient(config, "fake", transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(LLMError, match="llm_timeout"):
            await client.complete("Locate", {}, name="box")


async def test_provider_rejects_oversized_response_and_upload(monkeypatch):
    monkeypatch.setattr(part_grounding, "MAX_BYTES", 20)
    monkeypatch.setattr(part_grounding, "MAX_IMAGE_BYTES", 10)
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, content=b" " * 21)

    config = part_grounding.part_model_config()
    async with PartGroundingClient(config, "fake", transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(LLMError, match="part_image_too_large"):
            await client.complete("Locate", {}, name="box", image=Image.new("RGB", (4, 4)))
        assert not calls
        with pytest.raises(LLMError, match="llm_reply_too_large"):
            await client.complete("Locate", {}, name="box")
        assert len(calls) == 1


def test_refusal_and_invalid_box_are_rejected():
    refusal = {"type": "message", "role": "assistant", "content": [{"type": "refusal"}]}
    with pytest.raises(LLMError, match="llm_incomplete_reply"):
        reply_text({"status": "completed", "output": [refusal]}, "openai_responses")
    with pytest.raises(ValidationError):
        PartBox(box=[10, 0, 5, 3], status="matched", evidence="visible")
    with pytest.raises(ValidationError):
        PartBox(box=None, status="matched", evidence="visible")
    assert cli.check_box([0, 0, 1, 4], (8, 8), 2, 1) == "part_box_too_small"
    assert cli.check_box([0, 0, 8, 7], (8, 8), 2, 0.8) == "part_box_area_exceeded"
    expected = np.zeros((8, 8), dtype=bool)
    expected[1:3, 2:6] = True
    assert cli.expected_box_iou([2, 1, 6, 5], expected) == 0.5
    assert cli.expected_box_iou(None, expected) == 0


@pytest.mark.parametrize("ambiguous", [False, True])
def test_retry_records_crop_and_preserves_ambiguity(monkeypatch, ambiguous):
    fastsam = FastSamSegmenter.__new__(FastSamSegmenter)
    fastsam.confidence = 0.4
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    trace = []

    def masks(image, *, confidence):
        if image.shape == rgb.shape:
            raise ValueError("target_not_found")
        assert confidence == 0.4
        candidates = np.zeros((2 if ambiguous else 1, *image.shape[:2]), dtype=bool)
        candidates[:, 30:50, 40:60] = True
        return None, candidates

    monkeypatch.setattr(fastsam, "masks", masks)
    box = [150, 160, 170, 180]
    if ambiguous:
        with pytest.raises(ValueError, match="region_mask_ambiguous"):
            fastsam.segment_box_local_retry(rgb, box, trace=trace)
        assert trace[1]["reason"] == "region_mask_ambiguous"
    else:
        result, iou = fastsam.segment_box_local_retry(rgb, box, trace=trace)
        assert iou == 1 and result.sum() == 400 and result[160:180, 150:170].all()
    assert len(trace) == 2
    assert trace[0]["candidate_count"] == 0
    assert trace[1]["search_xyxy"] == [110, 130, 200, 200]


def test_exif_and_output_publication(tmp_path, monkeypatch):
    source = tmp_path / "source.jpg"
    image = Image.new("RGB", (40, 20))
    exif = image.getexif()
    exif[274] = 6
    image.save(source, exif=exif)
    assert cli.load_rgb(source).shape == (40, 20, 3)
    out = tmp_path / "run"
    args = ["--image", str(source), "--goal", "Lift the kettle", "--output-dir", str(out)]
    monkeypatch.setenv("OPENAI_API_KEY", "fake")

    async def fail(config, key, goal, image, staging):
        assert config.model == "gpt-6.2-sol" and config.reasoning_effort == "medium"
        (staging / "partial.json").write_text("{}")
        raise LLMError("llm_timeout")

    monkeypatch.setattr(cli, "locate_part", fail)
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 3 and not out.exists()
    assert not list(tmp_path.glob(".part-run-*"))

    box = tmp_path / "box.json"
    box.write_text(json.dumps({"box": None, "status": "part_not_visible", "evidence": "Occluded"}))
    with pytest.raises(SystemExit) as error:
        cli.main([*args, "--box-json", str(box)])
    assert error.value.code == 2
    result = json.loads(box.read_text())
    result["image"] = cli.image_identity(cli.load_rgb(source)).model_dump()
    box.write_text(json.dumps(result))
    assert cli.main([*args, "--box-json", str(box)]) == 1
    report = json.loads((out / "fastsam.json").read_text())
    assert report["status"] == "part_unverified" and report["replay_image_verified"] is True
    assert not (out / "input.png").exists()
    other = tmp_path / "other.png"
    Image.new("RGB", (20, 40), color="red").save(other)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        cli.replay_box(box, cli.image_identity(cli.load_rgb(other)), allow_legacy=False)


def test_annotation_binds_image_and_mask_bytes(tmp_path):
    identity = cli.image_identity(np.zeros((8, 10, 3), dtype=np.uint8))
    other = cli.image_identity(np.ones((8, 10, 3), dtype=np.uint8))
    path = tmp_path / "mask.png"
    Image.new("L", (10, 8), 255).save(path)
    annotation = {"image": identity.model_dump(),
                  "mask_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path.with_name("mask.png.json").write_text(json.dumps(annotation))
    assert cli.read_expected_mask(path, identity).all()
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        cli.read_expected_mask(path, other)
    Image.new("L", (10, 8), 128).save(path)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        cli.read_expected_mask(path, identity)


def test_model_settings_are_shared_and_credentials_are_endpoint_specific(tmp_path, monkeypatch):
    """A separate part provider must not forward the OpenAI key or use the planner model."""
    monkeypatch.setenv("LLM_MODEL", "unrelated-planning-model")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    config = part_grounding.part_model_config()
    assert config.model == "gpt-6.2-sol" and config.reasoning_effort == "medium"
    assert part_grounding.part_model_key(config) == "openai-test"

    env_file = tmp_path / "provider.env"
    env_file.write_text(
        "WRS_PART_LLM_BASE_URL=https://provider.example/v1\n"
        "WRS_PART_LLM_PROTOCOL=openai_chat\nWRS_PART_LLM_MODEL=vision-model\n"
    )
    settings = part_grounding.part_settings(env_file)
    config = part_grounding.part_model_config(settings, timeout_s=15)
    assert config.model == "vision-model" and config.reasoning_effort is None
    assert config.protocol == "openai_chat" and config.timeout_s == 15
    with pytest.raises(LLMError, match="llm_api_key_missing_or_invalid"):
        part_grounding.part_model_key(config, settings)
    assert part_grounding.part_model_key(config, {**settings, "LLM_API_KEY": "custom-test"}) == (
        "custom-test"
    )
    monkeypatch.setenv("LLM_API_KEY", "ordinary-planner-test")
    monkeypatch.setenv("LLM_BASE_URL", "https://other.example/v1")
    with pytest.raises(LLMError, match="llm_api_key_missing_or_invalid"):
        part_grounding.part_model_key(config, {
            **settings, "LLM_BASE_URL": "https://other.example/v1", "LLM_API_KEY": "file-test",
        })
    monkeypatch.setenv("WRS_PART_LLM_API_KEY", "part-only-test")
    assert part_grounding.part_model_key(config, settings) == "part-only-test"


async def test_two_api_phases_pass_through_non_handle_part():
    """Exercise the actual request serializers and response contracts for both stages."""
    model = part_grounding.part_model_config()
    calls = []
    goal = "Lift the box by its side panel"

    def handle(request):
        body = json.loads(request.content)
        content = body["input"][0]["content"]
        assert goal in content[0]["text"]
        calls.append(body["text"]["format"]["name"])
        if len(calls) == 1:
            assert len(content) == 1
            answer = {
                "object": "box", "action": "lift", "avoid_parts": [],
                "preferred_parts": [{
                    "part": "side panel", "reason": "Rigid panel", "condition": "Fits the jaws",
                }],
            }
        else:
            assert "Proposed part: side panel." in content[0]["text"]
            assert content[1]["type"] == "input_image"
            answer = {"box": [80, 100, 900, 600], "status": "matched", "evidence": "Visible panel."}
        return httpx.Response(200, json={"status": "completed", "output": [{
            "type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": json.dumps(answer)}],
        }]})

    async with PartGroundingClient(model, "fake", transport=httpx.MockTransport(handle)) as client:
        part = await client.suggest(goal)
        box = await client.ground(
            goal, part.object, part.preferred_parts[0].part, Image.new("RGB", (10, 10)),
        )
    assert calls == ["grasp_part_suggestion", "grasp_part_box"]
    assert box.box == [80, 100, 900, 600]
