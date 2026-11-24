"""Run language-guided FastSAM part segmentation on a saved RGB image."""

import argparse
import asyncio
import hashlib
import json
import struct
from importlib.resources import files
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from pydantic import Field, ValidationError

from wrs_agent.nodes.segmentation.fastsam import FastSamSegmenter
from wrs_agent.planner.grasp_part_prompts import PartBox, check_box
from wrs_agent.planner.providers.part_grounding import (
    MAX_IMAGE_BYTES,
    check_image_size,
    part_model_config,
    part_model_key,
    part_settings,
)
from wrs_agent.planner.providers.wire import LLMError
from wrs_agent.schemas import MAX_BYTES, Boundary

ROOT = Path(__file__).resolve().parents[1]


class ImageIdentity(Boundary):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class BoundPartBox(PartBox):
    image: ImageIdentity


class AnnotationIdentity(Boundary):
    image: ImageIdentity
    mask_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def image_identity(rgb):
    """Fingerprint dimensions and oriented pixels rather than the input file encoding."""
    height, width = rgb.shape[:2]
    digest = hashlib.sha256(struct.pack(">II", width, height))
    digest.update(memoryview(rgb).cast("B"))
    return ImageIdentity(width=width, height=height, sha256=digest.hexdigest())


def read_limited(path, limit):
    with path.open("rb") as source:
        data = source.read(limit + 1)
    if len(data) > limit:
        raise ValueError("local file exceeds the size limit")
    return data


def read_json_file(path):
    return read_limited(path, MAX_BYTES).decode("utf-8-sig")


def replay_box(path, identity, allow_legacy):
    """Accept legacy results only on explicit request, with unverified provenance."""
    raw = read_json_file(path)
    if allow_legacy and "image" not in json.loads(raw):
        return PartBox.model_validate_json(raw), False
    box = BoundPartBox.model_validate_json(raw)
    if box.image != identity:
        raise ValueError("replay image fingerprint mismatch")
    return box, True


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def model_settings(args, parser):
    try:
        settings = part_settings(args.env_file)
        config = part_model_config(
            settings,
            base_url=args.base_url, protocol=args.protocol, model=args.model,
            timeout_s=args.timeout, reasoning_effort=args.reasoning_effort,
            proxy=args.proxy, max_tokens=args.max_output_tokens,
        )
        return config, part_model_key(config, settings)
    except LLMError as exc:
        parser.error(f"model configuration failed: {exc.code}")


async def locate_part(config, api_key, goal, image, out):
    from wrs_agent.planner.providers.part_grounding import PartGroundingClient

    async with PartGroundingClient(config, api_key) as client:
        async with asyncio.timeout(config.timeout_s):
            suggestion = await client.suggest(goal)
            write_json(out / "grasp-part-suggestion.json", suggestion.model_dump())
            return await client.ground(goal, suggestion.object, suggestion.preferred_parts[0].part,
                                       image)


def load_rgb(path):
    with Image.open(BytesIO(read_limited(path, MAX_IMAGE_BYTES))) as source:
        check_image_size(*source.size)
        return np.array(ImageOps.exif_transpose(source).convert("RGB"))


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


def read_expected_mask(path, identity):
    annotation = AnnotationIdentity.model_validate_json(read_json_file(
        path.with_name(path.name + ".json"),
    ))
    raw = read_limited(path, MAX_IMAGE_BYTES)
    if (annotation.image != identity
            or hashlib.sha256(raw).hexdigest() != annotation.mask_sha256):
        raise ValueError("annotation fingerprint mismatch")
    with Image.open(BytesIO(raw)) as source:
        if source.size != (identity.width, identity.height):
            raise ValueError("expected mask must match the oriented image dimensions")
        expected = np.array(source.convert("L")) > 127
    if not expected.any():
        raise ValueError("expected mask must be nonempty")
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory")
    parser.add_argument("--model")
    parser.add_argument("--base-url")
    parser.add_argument("--protocol", choices=["openai_responses", "openai_chat"])
    parser.add_argument(
        "--reasoning-effort",
        choices=["omit", "none", "minimal", "low", "medium", "high", "xhigh", "max"],
    )
    parser.add_argument("--timeout", type=float, help="Total API budget in seconds")
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--proxy", help="Explicit HTTP(S) proxy")
    parser.add_argument("--env-file", type=Path, help="Read part-model settings and API credential")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--weights", type=Path, default=ROOT / ".local/models/fastsam/FastSAM-x.pt")
    parser.add_argument("--box-json", type=Path, help="Replay an existing gpt-box.json offline")
    parser.add_argument("--allow-legacy-box", action="store_true",
                        help="Allow an old unbound box without reporting verified provenance")
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
        args.model = config.model
    if args.allow_legacy_box and (not args.box_json or args.expected_mask):
        parser.error("--allow-legacy-box requires --box-json and cannot be used for evaluation")
    try:
        rgb = load_rgb(args.image)
        identity = image_identity(rgb)
        expected = read_expected_mask(args.expected_mask, identity) if args.expected_mask else None
        if args.box_json:
            answer, verified = replay_box(args.box_json, identity, args.allow_legacy_box)
    except (OSError, ValueError, Image.DecompressionBombError):
        parser.error("invalid image, annotation, or replay file (check size and fingerprints)")
    dest = args.output_dir.resolve()
    if dest.exists():
        parser.error("--output-dir already exists")
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Stage beside the destination so final publication is a same-volume rename.
    with TemporaryDirectory(dir=dest.parent, prefix=".part-run-") as staging:
        out = Path(staging) / "result"
        out.mkdir()
        try:
            if not args.box_json:
                answer = asyncio.run(locate_part(config, api_key, goal, Image.fromarray(rgb), out))
                verified = True
        except TimeoutError:
            parser.exit(3, "Model request failed: llm_timeout\n")
        except LLMError as exc:
            parser.exit(3, f"Model request failed: {exc.code}\n")
        except ValidationError:
            parser.exit(3, "Invalid structured grasp-part result\n")
        result = answer.model_dump(exclude={"image"})
        if verified:
            result["image"] = identity.model_dump()
        write_json(out / "gpt-box.json", result)
        write_json(out / "input-image.json", identity.model_dump())
        report, mask = segment_part(args, rgb, answer, out)
        report["replay_image_verified"] = verified if args.box_json else None
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
