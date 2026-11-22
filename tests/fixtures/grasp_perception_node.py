"""Offline model fixtures for the production SAM and GraspNet node lifecycles."""

import asyncio
import json
import sys
import time

import numpy as np

from wrs_agent.nodes.grasp import GraspNetNode
from wrs_agent.nodes.grasp.graspnet import candidates_from_rows
from wrs_agent.nodes.segmentation import Sam3Node
from wrs_agent.nodes.serve import serve_node


class FixtureSam:
    name, sha256 = "offline-sam-fixture", "0" * 64
    score_kind = "fixture"

    def segment(self, rgb, target, selection):
        time.sleep(.3)  # The camera's two-frame cache will have advanced during inference.
        if target == "ambiguous":
            raise ValueError("target_ambiguous")
        height, width = rgb.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        mask[height // 3:2 * height // 3, width // 3:2 * width // 3] = True
        return mask, .9

    def segment_box(self, rgb, box):
        time.sleep(.3)
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        x1, y1, x2, y2 = map(int, box)
        mask[y1:y2, x1:x2] = True
        return mask, .9

    def segment_box_local_retry(self, rgb, box):
        return self.segment_box(rgb, box)


class FixtureGrasp:
    name, sha256, revision = "offline-grasp-fixture", "0" * 64, "0" * 40

    def infer(self, points, mask):
        valid = mask & np.isfinite(points).all(axis=2) & (points[:, :, 2] > 0)
        if np.count_nonzero(valid) < 64:
            raise ValueError("insufficient_target_depth")
        center = points[valid].mean(axis=0)
        row = np.array([.9, .04, .02, .03, *np.eye(3).reshape(-1), *center, -1.])
        return candidates_from_rows(row[None], max_width_m=.05)


class FixtureSamNode(Sam3Node):
    def _part_client(self):
        import httpx

        from wrs_agent.planner.providers.part_grounding import (
            PartGroundingClient,
            part_model_config,
        )

        async def respond(request):
            body = json.loads(request.content)
            text, image = body["input"][0]["content"]
            assert "cup" in text["text"] and "handle" in text["text"]
            assert image["image_url"].startswith("data:image/png;base64,")
            assert body["model"] == "gpt-6.2-sol" and body["reasoning"]["effort"] == "medium"
            await asyncio.sleep(.3)
            if "missing handle" in text["text"]:
                result = {"box": None, "status": "part_not_visible", "evidence": "Occluded."}
            else:
                result = {"box": [85, 100, 190, 255], "status": "matched", "evidence": "Visible."}
            return httpx.Response(200, json={"status": "completed", "output": [{
                "type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": json.dumps(result)}],
            }]})

        return PartGroundingClient(part_model_config(), "offline-fixture",
                                   transport=httpx.MockTransport(respond))

    def _segmenter(self):
        model = FixtureSam()
        if self.options.backend == "fastsam":
            model.name, model.score_kind = "offline-clip-fixture", "clip_cosine"
            model.text_sha256 = "1" * 64
        if self.options.backend in {"fastsam-box", "fastsam-part"}:
            model.name, model.score_kind = "offline-box-fixture", "box_iou"
        return model


class FixtureGraspNode(GraspNetNode):
    def _predictor(self):
        return FixtureGrasp()


if __name__ == "__main__":
    kind, endpoint, env_id = sys.argv[1:4]
    options = {"weights": "unused-fixture.pt", "weights_sha256": "0" * 64}
    if kind == "graspnet":
        options.update(source="unused-fixture", source_revision="0" * 40,
                       segmentation=sys.argv[4] if len(sys.argv) > 4 else "sam3")
    if kind == "segmentation":
        options.update(backend="fastsam", clip_weights="unused-clip-fixture.pt")
    if kind == "segmentation-box":
        options.update(backend="fastsam-box")
    if kind == "segmentation-part":
        options.update(backend="fastsam-part")
    asyncio.run(serve_node(FixtureGraspNode if kind == "graspnet" else FixtureSamNode,
                           node_id=kind, endpoint=endpoint, env_id=env_id, options=options))
