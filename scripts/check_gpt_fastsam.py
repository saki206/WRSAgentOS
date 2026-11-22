"""Run language-guided FastSAM part segmentation on a saved RGB image."""

import argparse
import asyncio
import json
import os
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "gpt-6.2-sol"
DEFAULT_URL = "https://api.openai.com/v1"


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def model_settings(args, parser):
    from wrs_agent.planner.providers.llm import LLMConfig

    settings = {}
    if args.env_file is not None:
        from dotenv import dotenv_values

        if not args.env_file.is_file():
            parser.error("--env-file must point to an existing file")
        settings = dotenv_values(args.env_file, encoding="utf-8-sig", interpolate=False)
    effort = args.reasoning_effort
    if effort is None and args.model == DEFAULT_MODEL:
        effort = "medium"
    try:
        config = LLMConfig(
            base_url=args.base_url, protocol=args.protocol, model=args.model,
            timeout_s=args.timeout, reasoning_effort=None if effort == "omit" else effort,
            proxy=args.proxy, max_tokens=args.max_output_tokens,
        )
    except ValidationError as exc:
        field = exc.errors(include_input=False)[0]["loc"][0]
        parser.error(f"invalid model configuration: {field}")
    # Custom endpoints receive only their explicitly configured provider credential.
    key_name = "OPENAI_API_KEY" if config.base_url == DEFAULT_URL else "LLM_API_KEY"
    key = os.environ.get(key_name) or settings.get(key_name)
    if not key and config.local:
        return config, ""
    if not key or not key.isascii() or not key.isprintable() or " " in key:
        parser.error(f"set {key_name} or provide it via --env-file")
    return config, key


async def locate_part(config, api_key, goal, image, out):
    from wrs_agent.planner.providers.part_grounding import PartGroundingClient

    async with PartGroundingClient(config, api_key) as client:
        async with asyncio.timeout(config.timeout_s):
            suggestion = GraspPartSuggestion.model_validate_json(await client.complete(
                f"{SYSTEM_PROMPT}\nUser command: {goal}", PART_SCHEMA, name="grasp_part_suggestion",
            ))
            write_json(out / "grasp-part-suggestion.json", suggestion.model_dump())
            prompt = grounding_prompt(goal, suggestion.object, suggestion.preferred_parts[0].part)
            return PartBox.model_validate_json(await client.complete(
                prompt, PartBox.model_json_schema(), name="grasp_part_box", image=image,
            ))


def load_rgb(path):
    with Image.open(path) as source:
        return np.array(ImageOps.exif_transpose(source).convert("RGB"))


def check_box(box, shape, minimum, max_fraction):
    x1, y1, x2, y2 = box
    if min(x2 - x1, y2 - y1) < minimum:
        return "part_box_too_small"
    if (x2 - x1) * (y2 - y1) > max_fraction * shape[0] * shape[1]:
        return "part_box_area_exceeded"
    return None


def save_overlay(rgb, box, mask, path):
    pixels = rgb.copy()
    if mask is not None:
        pixels[mask] = (0.45 * pixels[mask] + 0.55 * np.array([255, 50, 20])).astype(np.uint8)
    image = Image.fromarray(pixels)
    if box is not None:
        ImageDraw.Draw(image).rectangle(box, outline=(0, 255, 255), width=3)
    image.save(path, quality=94)


def segment_part(args, rgb, answer, out):
    height, width = rgb.shape[:2]
    box = answer.pixel_box(width, height)
    report = {
        "model": None if args.box_json else args.model,
        "box_source": "replay" if args.box_json else "api",
        "grounding_status": answer.status, "pixel_box": box, "mask": None,
        "box_iou": None, "attempts": [], "box_iou_minimum": 0.5, "box_iou_margin": 0.05,
    }
    mask = None
    if box is None:
        report["status"] = "part_unverified"
    elif reason := check_box(box, rgb.shape, args.min_box_pixels, args.max_box_fraction):
        report.update(status="box_rejected", error=reason)
    else:
        manifest = files("wrs_agent.nodes.segmentation").joinpath("fastsam_assets.json")
        sha256 = json.loads(manifest.read_text("utf-8"))["sha256"]
        fastsam = FastSamSegmenter(args.weights, sha256, device=args.device)
        try:
            mask, score = fastsam.segment_box_local_retry(rgb, box, trace=report["attempts"])
        except ValueError as exc:
            report.update(status="mask_selection_failed", error=str(exc))
        else:
            mask_name = "fastsam-final-mask.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(out / mask_name)
            report.update(status="matched", mask=mask_name, box_iou=score)
    return report, mask


def read_expected_mask(path, shape):
    with Image.open(path) as source:
        expected = np.array(source.convert("L")) > 127
    if expected.shape != shape or not expected.any():
        raise ValueError("expected mask must be nonempty and match the oriented image dimensions")
    return expected


def expected_box_iou(box, mask):
    if box is None:
        return 0.0
    rows, columns = np.nonzero(mask)
    left, top, right, bottom = columns.min(), rows.min(), columns.max() + 1, rows.max() + 1
    x1, y1, x2, y2 = box
    overlap = max(0, min(x2, right) - max(x1, left)) * max(0, min(y2, bottom) - max(y1, top))
    union = (x2 - x1) * (y2 - y1) + (right - left) * (bottom - top) - overlap
    return float(overlap / union)


def main(argv=None):
    from wrs_agent.planner.providers.wire import LLMError

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_URL)
    parser.add_argument("--protocol", choices=["openai_responses", "openai_chat"],
                        default="openai_responses")
    parser.add_argument(
        "--reasoning-effort",
        choices=["omit", "none", "minimal", "low", "medium", "high", "xhigh", "max"],
    )
    parser.add_argument("--timeout", type=float, default=120, help="Total API budget in seconds")
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--proxy", help="Explicit HTTP(S) proxy")
    parser.add_argument("--env-file", type=Path, help="Read only the selected API credential")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--weights", type=Path, default=ROOT / ".local/models/fastsam/FastSAM-x.pt")
    parser.add_argument("--box-json", type=Path, help="Replay an existing gpt-box.json offline")
    parser.add_argument("--min-box-pixels", type=float, default=2)
    parser.add_argument("--max-box-fraction", type=float, default=1)
    parser.add_argument("--expected-mask", type=Path, help="Manual mask of the visible target part")
    parser.add_argument(
        "--save-images", action="store_true", help="Save input RGB and color overlay",
    )
    args = parser.parse_args(argv)
    goal = args.goal.strip()
    if not goal or len(goal) > 1024:
        parser.error("--goal must contain 1 to 1024 characters")
    if not 1 <= args.min_box_pixels <= 10000 or not 0 < args.max_box_fraction <= 1:
        parser.error("invalid box size limits")
    if args.box_json is None:
        config, api_key = model_settings(args, parser)
    rgb = load_rgb(args.image)
    expected = read_expected_mask(args.expected_mask, rgb.shape[:2]) if args.expected_mask else None
    dest = args.output_dir.resolve()
    if dest.exists():
        parser.error("--output-dir already exists")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=dest.parent, prefix=".part-run-") as staging:
        out = Path(staging) / "result"
        out.mkdir()
        try:
            if args.box_json:
                answer = PartBox.model_validate_json(args.box_json.read_text(encoding="utf-8-sig"))
            else:
                answer = asyncio.run(locate_part(config, api_key, goal, Image.fromarray(rgb), out))
        except TimeoutError:
            parser.exit(3, "Model request failed: llm_timeout\n")
        except LLMError as exc:
            parser.exit(3, f"Model request failed: {exc.code}\n")
        except ValidationError:
            parser.exit(3, "Invalid structured grasp-part result\n")
        write_json(out / "gpt-box.json", answer.model_dump())
        report, mask = segment_part(args, rgb, answer, out)
        if expected is not None:
            report["expected_box_iou"] = expected_box_iou(report["pixel_box"], expected)
            report["expected_mask_iou"] = (
                float((mask & expected).sum() / (mask | expected).sum())
                if mask is not None else 0.0
            )
        if args.save_images:
            Image.fromarray(rgb).save(out / "input.png")
            save_overlay(rgb, report["pixel_box"], mask, out / "fastsam-final-overlay.jpg")
        write_json(out / "fastsam.json", report)
        if dest.exists():
            parser.error("--output-dir already exists")
        out.rename(dest)
    print(f"{report['status']} · {dest}")
    return 0 if mask is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
