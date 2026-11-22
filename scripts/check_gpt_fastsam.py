"""Run language-guided FastSAM part segmentation on a saved RGB image."""

import argparse
import base64
import json
import os
from importlib.resources import files
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from pydantic import ValidationError

from wrs_agent.nodes.segmentation.fastsam import FastSamSegmenter
from wrs_agent.planner.grasp_part_prompts import (
    PART_SCHEMA,
    SYSTEM_PROMPT,
    GraspPartSuggestion,
    PartBox,
    grounding_prompt,
)
from wrs_agent.planner.providers import openai_responses
from wrs_agent.planner.providers.wire import LLMError

ROOT = Path(__file__).resolve().parents[1]
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_gpt_json(response):
    try:
        body = response.json()
        if body["status"] != "completed":
            raise LLMError("llm_incomplete_reply")
        messages = [item for item in body["output"] if item["type"] == "message"]
        message, = messages
        parts = message["content"]
        if message["role"] != "assistant" or any(part["type"] != "output_text" for part in parts):
            raise LLMError("llm_invalid_reply")
        return "".join(part["text"] for part in parts)
    except (KeyError, TypeError, ValueError):
        raise LLMError("llm_invalid_reply") from None


def request_gpt(client, model, prompt, schema, *, name, image=None):
    import httpx

    content = [{"type": "input_text", "text": prompt}]
    if image is not None:
        encoded = base64.b64encode(image.read_bytes()).decode("ascii")
        content.append({
            "type": "input_image", "image_url": f"data:image/png;base64,{encoded}",
            "detail": "high",
        })
    try:
        response = client.post(OPENAI_RESPONSES_URL, json={
            "model": model,
            "store": False,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": 2048,
            "text": {"format": {
                "type": "json_schema", "name": name, "strict": True, "schema": schema,
            }},
        })
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise SystemExit(f"OpenAI API returned HTTP {exc.response.status_code} ({name}).") from None
    except httpx.RequestError:
        raise SystemExit(f"OpenAI API request failed ({name}). Check the connection.") from None
    return parse_gpt_json(response)


def locate_with_gpt(api_key, model, goal, out):
    import httpx

    with httpx.Client(headers=openai_responses.headers(api_key), timeout=120) as client:
        suggestion = GraspPartSuggestion.model_validate_json(request_gpt(
            client, model, f"{SYSTEM_PROMPT}\nUser command: {goal}",
            PART_SCHEMA, name="grasp_part_suggestion",
        ))
        write_json(out / "grasp-part-suggestion.json", suggestion.model_dump())
        prompt = grounding_prompt(goal, suggestion.object, suggestion.preferred_parts[0].part)
        (out / "grounding-prompt.txt").write_text(prompt, encoding="utf-8")
        return PartBox.model_validate_json(request_gpt(
            client, model, prompt, PartBox.model_json_schema(),
            name="grasp_part_box", image=out / "input.png",
        ))


def save_overlay(rgb, box, mask, path):
    pixels = rgb.copy()
    if mask is not None:
        pixels[mask] = (0.45 * pixels[mask] + 0.55 * np.array([255, 50, 20])).astype(np.uint8)
    image = Image.fromarray(pixels)
    if box is not None:
        ImageDraw.Draw(image).rectangle(box, outline=(0, 255, 255), width=3)
    image.save(path, quality=94)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI API vision model")
    parser.add_argument("--env-file", type=Path, help="Load OPENAI_API_KEY from a local .env file")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--weights", type=Path, default=ROOT / ".local/models/fastsam/FastSAM-x.pt")
    parser.add_argument("--box-json", type=Path, help="Replay an existing gpt-box.json offline")
    args = parser.parse_args()
    goal = args.goal.strip()
    if not goal or len(goal) > 1024:
        parser.error("--goal must contain 1 to 1024 characters")
    if args.box_json is None:
        if args.env_file is not None:
            from dotenv import dotenv_values

            if not args.env_file.is_file():
                parser.error("--env-file must point to an existing file")
            settings = dotenv_values(args.env_file, encoding="utf-8-sig", interpolate=False)
        else:
            settings = {}
        api_key = os.environ.get("OPENAI_API_KEY") or settings.get("OPENAI_API_KEY")
        if not api_key or not api_key.isascii() or not api_key.isprintable() or " " in api_key:
            parser.error("set OPENAI_API_KEY or provide it via --env-file")
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    rgb = np.array(image)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    image.save(out / "input.png")

    if args.box_json:
        answer = PartBox.model_validate_json(args.box_json.read_text(encoding="utf-8-sig"))
    else:
        try:
            answer = locate_with_gpt(api_key, args.model, goal, out)
        except (LLMError, ValidationError):
            raise SystemExit("OpenAI API did not return a valid grasp-part result.") from None
    write_json(out / "gpt-box.json", answer.model_dump())

    box = answer.pixel_box(image.width, image.height)
    report = {"goal": goal, "box_source": "replay" if args.box_json else args.model,
              "grounding_status": answer.status, "pixel_box": box,
              "mask": None, "box_iou": None}
    mask = None
    if box is None:
        report["status"] = "part_unverified"
    else:
        manifest = files("wrs_agent.nodes.segmentation").joinpath("fastsam_assets.json")
        sha256 = json.loads(manifest.read_text("utf-8"))["sha256"]
        fastsam = FastSamSegmenter(args.weights, sha256, device=args.device)
        try:
            mask, score = fastsam.segment_box_local_retry(rgb, box)
        except ValueError as exc:
            report.update(status="mask_selection_failed", error=str(exc))
        else:
            mask_name = "fastsam-final-mask.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(out / mask_name)
            report.update(status="matched", mask=mask_name, box_iou=score)

    save_overlay(rgb, box, mask, out / "fastsam-final-overlay.jpg")
    write_json(out / "fastsam.json", report)
    print(f"{report['status']} · {out}")
    return 0 if mask is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
