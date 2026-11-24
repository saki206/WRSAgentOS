"""部位定位和 FastSAM 的离线回归，使用模拟服务和小幅测试图片。"""

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
    """检查两种协议各自的鉴权、图像和 JSON Schema 请求格式。"""
    config = LLMConfig(base_url=part_grounding.DEFAULT_URL, model=part_grounding.DEFAULT_MODEL,
                       protocol=protocol, reasoning_effort="medium")

    def handle(request):
        # 拦截真实客户端产生的 HTTP 请求，再返回该协议的模拟响应。
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
    """模拟响应比配置时限更慢，确认程序返回统一的超时错误。"""
    async def handle(_request):
        await asyncio.sleep(1)
        return httpx.Response(500)

    config = LLMConfig(base_url=part_grounding.DEFAULT_URL, model=part_grounding.DEFAULT_MODEL,
                       protocol="openai_responses", timeout_s=0.01)
    async with PartGroundingClient(config, "fake", transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(LLMError, match="llm_timeout"):
            await client.complete("Locate", {}, name="box")


async def test_provider_rejects_oversized_response_and_upload(monkeypatch):
    """用调小的字节上限覆盖上传拦截和响应中途终止。"""
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
    """校验拒答、非法坐标、空框状态冲突及几何限制。"""
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
    """整图无候选时检查裁剪重试、掩码还原和歧义退出。"""
    fastsam = FastSamSegmenter.__new__(FastSamSegmenter)
    fastsam.confidence = 0.4
    rgb = np.zeros((200, 200, 3), dtype=np.uint8)
    trace = []

    def masks(image, *, confidence):
        # 模拟裁剪内只出现一个掩码，或出现两个同样匹配的掩码。
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
    """检查 EXIF 方向、异常清理及绑定图片后的离线复放。"""
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
    # 模拟接口写出中间结果后失败，最终输出目录应仍未发布。
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 3 and not out.exists()
    assert not list(tmp_path.glob(".part-run-*"))

    box = tmp_path / "box.json"
    # 同一份框分别验证缺少指纹、绑定正确原图和替换成同尺寸图片的情况。
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
    """替换原图或篡改掩码文件后，评估入口都应拒绝。"""
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
    """更换部位服务时，检查其配置与任务规划配置及 OpenAI 凭据隔离。"""
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


async def test_two_api_phases_pass_through_non_handle_part():
    """用方盒侧板验证两阶段动态传参，覆盖把手以外的部位。"""
    model = part_grounding.part_model_config()
    calls = []
    goal = "Lift the box by its side panel"

    def handle(request):
        body = json.loads(request.content)
        content = body["input"][0]["content"]
        assert goal in content[0]["text"]
        calls.append(body["text"]["format"]["name"])
        if len(calls) == 1:
            # 第一段只看口令。第二段必须收到第一段返回的部位和待定位图片。
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
