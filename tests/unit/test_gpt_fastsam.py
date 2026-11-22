import asyncio
import hashlib
import json

import httpx
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from scripts import check_gpt_fastsam as cli
from wrs_agent.nodes.segmentation.fastsam import FastSamSegmenter
from wrs_agent.planner.grasp_part_prompts import PartBox
from wrs_agent.planner.providers import part_grounding
from wrs_agent.planner.providers.llm import LLMConfig
from wrs_agent.planner.providers.part_grounding import PartGroundingClient, reply_text
from wrs_agent.planner.providers.wire import LLMError


@pytest.mark.parametrize("protocol", ["openai_responses", "openai_chat"])
async def test_provider_request_with_image(protocol):
    config = LLMConfig(base_url=cli.DEFAULT_URL, model=cli.DEFAULT_MODEL,
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

    config = LLMConfig(base_url=cli.DEFAULT_URL, model=cli.DEFAULT_MODEL,
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
