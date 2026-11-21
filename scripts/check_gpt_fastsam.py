"""Run language-guided FastSAM part segmentation on a saved RGB image."""

import argparse
import json
import subprocess
from importlib.resources import files
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from wrs_agent.nodes.segmentation.fastsam import FastSamSegmenter
from wrs_agent.planner.grasp_part_prompts import (
    PART_SCHEMA,
    SYSTEM_PROMPT,
    GraspPartSuggestion,
    PartBox,
    grounding_prompt,
)

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_codex(model, prompt, schema, output, *, image=None):
    schema_file = output.with_suffix(".schema.json")
    write_json(schema_file, schema)
    command = [
        "codex", "exec", "--ephemeral", "--sandbox", "read-only", "--ignore-user-config",
        "--skip-git-repo-check", "--output-schema", str(schema_file),
        "--output-last-message", str(output), "-m", model, "-C", str(output.parent),
    ]
    if image is not None:
        command.extend(["--image", str(image)])
    command += ["--", "-"]
    prompt += "\nUse only the supplied text and image. Do not call tools or read files."
    subprocess.run(command, input=prompt, encoding="utf-8", check=True, timeout=300)
    return output.read_text(encoding="utf-8")


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
    parser.add_argument("--model", default="gpt-6.1-sol", help="Model available in Codex")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--weights", type=Path, default=ROOT / ".local/models/fastsam/FastSAM-x.pt")
    parser.add_argument("--box-json", type=Path, help="Replay an existing gpt-box.json offline")
    args = parser.parse_args()
    goal = args.goal.strip()
    if not goal or len(goal) > 1024:
        parser.error("--goal must contain 1 to 1024 characters")
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    rgb = np.array(image)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    image.save(out / "input.png")

    if args.box_json:
        answer = PartBox.model_validate_json(args.box_json.read_text(encoding="utf-8-sig"))
    else:
        suggestion = GraspPartSuggestion.model_validate_json(run_codex(
            args.model,
            f"{SYSTEM_PROMPT}\nUser command: {goal}\nReturn only the specified JSON.",
            PART_SCHEMA,
            out / "grasp-part-suggestion.json",
        ))
        part = suggestion.preferred_parts[0].part
        prompt = grounding_prompt(goal, suggestion.object, part)
        (out / "grounding-prompt.txt").write_text(prompt, encoding="utf-8")
        answer = PartBox.model_validate_json(run_codex(
            args.model, prompt, PartBox.model_json_schema(), out / "gpt-box.json",
            image=out / "input.png",
        ))
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
