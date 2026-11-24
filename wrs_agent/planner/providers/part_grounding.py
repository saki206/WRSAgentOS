"""Structured text and image requests for grasp part selection."""

import asyncio
import base64
import json
import os
from io import BytesIO
from pathlib import Path

import httpx
from PIL import Image

from wrs_agent.planner.grasp_part_prompts import (
    PART_SCHEMA,
    SYSTEM_PROMPT,
    GraspPartSuggestion,
    PartBox,
    grounding_prompt,
)
from wrs_agent.planner.providers import openai_chat, openai_responses
from wrs_agent.planner.providers.llm import LLMConfig, http_transport, transport_error_code
from wrs_agent.planner.providers.wire import LLMError
from wrs_agent.schemas import MAX_BYTES

PROTOCOLS = {"openai_responses": openai_responses, "openai_chat": openai_chat}
DEFAULT_MODEL = "gpt-6.2-sol"
DEFAULT_URL = "https://api.openai.com/v1"
MAX_IMAGE_PIXELS = 4096 * 4096
MAX_IMAGE_BYTES = 16 * 1024 * 1024


def part_settings(env_file=None):
    """Read only part-specific settings so the robot planner can use a different model."""
    if env_file is None:
        return {}
    from dotenv import dotenv_values

    if not Path(env_file).is_file():
        raise LLMError("part_env_file_not_found")
    try:
        return dotenv_values(env_file, encoding="utf-8-sig", interpolate=False)
    except (OSError, UnicodeError):
        raise LLMError("part_env_file_invalid") from None


def part_model_config(settings=None, **overrides):
    """Use repository model validation with defaults shared by both entry points."""
    settings = settings or {}
    defaults = {
        "base_url": DEFAULT_URL, "protocol": "openai_responses", "model": DEFAULT_MODEL,
        "timeout_s": "120", "max_tokens": "4096", "proxy": "",
    }
    selected = {}
    for name, default in defaults.items():
        variable = "WRS_PART_LLM_" + name.upper()
        selected["LLM_" + name.upper()] = (
            os.environ.get(variable) or settings.get(variable) or default
        )
    effort_variable = "WRS_PART_LLM_REASONING_EFFORT"
    effort = os.environ.get(effort_variable) or settings.get(effort_variable)
    for name, value in overrides.items():
        if value is not None and name != "reasoning_effort":
            selected["LLM_" + name.upper()] = str(value)
    effort = overrides.get("reasoning_effort") or effort
    if effort is None and selected["LLM_MODEL"].strip() == DEFAULT_MODEL:
        effort = "medium"
    selected["LLM_REASONING_EFFORT"] = "" if effort == "omit" else effort
    config = LLMConfig.from_env(selected)
    if config.protocol not in PROTOCOLS:
        raise LLMError("part_protocol_unsupported")
    return config


def part_model_key(config, settings=None):
    """Send OpenAI credentials only to the default OpenAI endpoint."""
    key_name = "OPENAI_API_KEY" if config.base_url == DEFAULT_URL else "LLM_API_KEY"
    key = os.environ.get(key_name) or (settings or {}).get(key_name)
    if not key or not key.isascii() or not key.isprintable() or " " in key:
        raise LLMError("llm_api_key_missing_or_invalid")
    return key


def check_image_size(width, height):
    if not 0 < width * height <= MAX_IMAGE_PIXELS:
        raise LLMError("part_image_too_large")


def encode_image(image: Image.Image):
    # Rebuild from RGB bytes so EXIF and other source metadata stay out of the request.
    check_image_size(*image.size)
    buffer = BytesIO()
    rgb = image.convert("RGB")
    Image.frombytes("RGB", rgb.size, rgb.tobytes()).save(buffer, format="PNG")
    if buffer.tell() > MAX_IMAGE_BYTES:
        raise LLMError("part_image_too_large")
    return "data:image/png;base64," + base64.b64encode(buffer.getbuffer()).decode("ascii")


def reply_text(body, protocol):
    try:
        if protocol == "openai_chat":
            choice, = body["choices"]
            message = choice["message"]
            if (choice["finish_reason"] != "stop" or message.get("refusal")
                    or message.get("tool_calls") or message.get("function_call")):
                raise LLMError("llm_incomplete_reply")
            text = message["content"]
        else:
            if body["status"] != "completed":
                raise LLMError("llm_incomplete_reply")
            output = body["output"]
            if any(item["type"] not in {"reasoning", "message"} for item in output):
                raise LLMError("llm_invalid_reply")
            message, = (item for item in output if item["type"] == "message")
            parts = message["content"]
            if any(part["type"] != "output_text" for part in parts):
                raise LLMError("llm_incomplete_reply")
            text = "".join(part["text"] for part in parts)
        if message["role"] != "assistant" or not isinstance(text, str) or not text:
            raise ValueError("missing_assistant_text")
        return text
    except LLMError:
        raise
    except (KeyError, TypeError, ValueError):
        raise LLMError("llm_invalid_reply") from None


class PartGroundingClient:
    """Key-authenticated HTTP client for both text suggestions and image localization."""
    def __init__(self, config, key, *, transport=None):
        self.config = config
        self.wire = PROTOCOLS[config.protocol]
        self.client = httpx.AsyncClient(
            base_url=config.base_url + "/",
            headers=self.wire.headers(key),
            timeout=config.timeout_s,
            trust_env=False,
            transport=transport if transport is not None else http_transport(config),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()

    async def aclose(self):
        await self.client.aclose()

    async def complete(self, prompt, schema, *, name, image=None):
        config = self.config
        responses = config.protocol == "openai_responses"
        content = [{"type": "input_text" if responses else "text", "text": prompt}]
        if image is not None:
            url = await asyncio.to_thread(encode_image, image)
            if responses:
                content.append({"type": "input_image", "image_url": url, "detail": "high"})
            else:
                content.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
        structured = {"name": name, "strict": True, "schema": schema}
        messages = [{"role": "user", "content": content}]
        body = {"model": config.model, "store": False}
        if responses:
            body.update(input=messages, max_output_tokens=config.max_tokens,
                        text={"format": {"type": "json_schema", **structured}})
            if config.reasoning_effort is not None:
                body["reasoning"] = {"effort": config.reasoning_effort}
        else:
            body.update(messages=messages, max_completion_tokens=config.max_tokens,
                        response_format={"type": "json_schema", "json_schema": structured})
            if config.reasoning_effort is not None:
                body["reasoning_effort"] = config.reasoning_effort
        try:
            async with asyncio.timeout(config.timeout_s):
                # Bound bytes as they arrive and only decode a complete, small response.
                data = bytearray()
                async with self.client.stream("POST", self.wire.PATH, json=body) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        if len(data) + len(chunk) > MAX_BYTES:
                            raise LLMError("llm_reply_too_large")
                        data.extend(chunk)
                return reply_text(json.loads(data), config.protocol)
        except httpx.HTTPStatusError as exc:
            raise LLMError(f"llm_http_{exc.response.status_code}") from None
        except (TimeoutError, httpx.TimeoutException):
            raise LLMError("llm_timeout") from None
        except httpx.RequestError as exc:
            raise LLMError(transport_error_code(exc)) from None
        except LLMError:
            raise
        except ValueError:
            raise LLMError("llm_invalid_reply") from None

    async def suggest(self, goal):
        return GraspPartSuggestion.model_validate_json(await self.complete(
            f"{SYSTEM_PROMPT}\nUser command: {goal}", PART_SCHEMA, name="grasp_part_suggestion",
        ))

    async def ground(self, goal, target, part, image, *, object_box=None):
        prompt = grounding_prompt(goal, target, part)
        if object_box is not None:
            width, height = image.size
            xyxy = [round(value / size * 999) for value, size in
                    zip(object_box, (width, height, width, height), strict=True)]
            prompt += (
                f" The detector selected the target instance at normalized box {xyxy}."
                " Use it to disambiguate instances. Attached parts may protrude outside that box."
            )
        return PartBox.model_validate_json(await self.complete(
            prompt, PartBox.model_json_schema(), name="grasp_part_box", image=image,
        ))
