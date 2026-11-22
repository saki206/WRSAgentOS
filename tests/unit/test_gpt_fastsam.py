import asyncio
import json

import httpx
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from scripts import check_gpt_fastsam as cli
from wrs_agent.nodes.segmentation.fastsam import FastSamSegmenter
from wrs_agent.planner.grasp_part_prompts import PartBox
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
    assert cli.main([*args, "--box-json", str(box)]) == 1
    assert json.loads((out / "fastsam.json").read_text())["status"] == "part_unverified"
    assert not (out / "input.png").exists()
