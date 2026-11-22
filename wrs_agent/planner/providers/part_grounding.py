"""Structured text and image requests for grasp part selection."""

import asyncio
import base64
from io import BytesIO

import httpx

from wrs_agent.planner.providers import openai_chat, openai_responses
from wrs_agent.planner.providers.llm import http_transport, transport_error_code
from wrs_agent.planner.providers.wire import LLMError

PROTOCOLS = {"openai_responses": openai_responses, "openai_chat": openai_chat}


def reply_text(body, protocol):
    try:
        if protocol == "openai_chat":
            choice, = body["choices"]
            message = choice["message"]
            if choice["finish_reason"] != "stop" or message.get("refusal"):
                raise LLMError("llm_incomplete_reply")
            text = message["content"]
        else:
            if body["status"] != "completed":
                raise LLMError("llm_incomplete_reply")
            message, = (item for item in body["output"] if item["type"] == "message")
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
        await self.client.aclose()

    async def complete(self, prompt, schema, *, name, image=None):
        config = self.config
        responses = config.protocol == "openai_responses"
        content = [{"type": "input_text" if responses else "text", "text": prompt}]
        if image is not None:
            data = BytesIO()
            image.save(data, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(data.getvalue()).decode("ascii")
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
                response = await self.client.post(self.wire.PATH, json=body)
                response.raise_for_status()
                return reply_text(response.json(), config.protocol)
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
