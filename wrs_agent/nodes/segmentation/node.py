"""One segmentation node lifecycle, with explicitly selected resident model backends."""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from wrs_agent.errors import AgentError, error_info, known_error_info
from wrs_agent.executor import ActionExecutor, SkillFailure
from wrs_agent.nodes.camera.client import CameraClient
from wrs_agent.nodes.node import Node
from wrs_agent.nodes.segmentation.data import (
    LatestSegment,
    PartProvenance,
    SegmentData,
    SegmentInfo,
    pack_segment,
)
from wrs_agent.nodes.vision.artifacts import ArtifactCache, ArtifactRequest, artifact_owner
from wrs_agent.nodes.vision.locations import read_location
from wrs_agent.schemas import Boundary, Empty, Name, VisionData, new_id
from wrs_agent.skills.vision.grasping import (
    SEGMENT,
    SEGMENT_PART,
    SEGMENT_REGION,
    PartRegionArgs,
    SegmentOutput,
)

BOX_BACKENDS = {"fastsam-box", "fastsam-part"}


def part_box_overlaps_object(part_box, object_box):
    """Check the projected attachment area without constraining long parts' centers."""
    px1, py1, px2, py2 = part_box
    ox1, oy1, ox2, oy2 = object_box
    overlap_width = min(px2, ox2) - max(px1, ox1)
    overlap_height = min(py2, oy2) - max(py1, oy1)
    # Ignore edge contact and small slivers while allowing most of a long handle outside.
    min_width = max(2, .1 * min(px2 - px1, ox2 - ox1))
    min_height = max(2, .1 * min(py2 - py1, oy2 - oy1))
    return overlap_width >= min_width and overlap_height >= min_height


class SegmentationOptions(Boundary):
    backend: Literal["sam3", "fastsam", "fastsam-box", "fastsam-part"] = "sam3"
    weights: Path = Field(strict=False)
    weights_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    device: Literal["cpu", "cuda"] = "cuda"
    confidence: float = Field(default=0.5, gt=0, le=1)
    camera: Name = "camera"
    locator: Name = "yolo"
    clip_weights: Path | None = Field(default=None, strict=False)
    text_similarity: float = Field(default=.28, gt=0, le=1)
    text_margin: float = Field(default=.03, ge=0, le=1)

    @model_validator(mode="after")
    def check_backend(self):
        if self.backend == "fastsam" and self.clip_weights is None:
            raise ValueError("fastsam_requires_local_clip_weights")
        if self.backend != "fastsam" and self.clip_weights is not None:
            raise ValueError("clip_weights_require_fastsam_backend")
        return self


@dataclass
class SegmentationState:
    version: int = 0
    completed: int = 0

    def snapshot(self):
        return VisionData(completed=self.completed, health="ready")


class SegmentationNode(Node):
    node_type = "vision"
    options_type = SegmentationOptions
    action_service = True

    def _segmenter(self):
        if self.options.backend in BOX_BACKENDS | {"fastsam"}:
            from wrs_agent.nodes.segmentation.fastsam import FastSamSegmenter

            return FastSamSegmenter(
                self.options.weights, self.options.weights_sha256,
                clip_weights=self.options.clip_weights, device=self.options.device,
                confidence=self.options.confidence, minimum=self.options.text_similarity,
                margin=self.options.text_margin,
            )
        from wrs_agent.nodes.segmentation.sam3 import Sam3Segmenter

        return Sam3Segmenter(self.options.weights, self.options.weights_sha256,
                             device=self.options.device, confidence=self.options.confidence)

    def _part_client(self):
        from wrs_agent.planner.providers.part_grounding import (
            PartGroundingClient,
            part_model_config,
            part_model_key,
            part_settings,
        )

        settings = part_settings(os.environ.get("WRS_PART_LLM_ENV_FILE"))
        config = part_model_config(settings)
        return PartGroundingClient(config, part_model_key(config, settings))

    async def setup(self):
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="segmentation")
        self.on_close(self._close_worker)
        self.loop = asyncio.get_running_loop()
        self.segmenter = await self.loop.run_in_executor(self.worker, self._segmenter)
        self.cache = ArtifactCache()
        self.latest_request = None
        if self.options.backend == "fastsam-part":
            self.part_client = self._part_client()
            self.on_close(self.part_client.aclose)
        skills = {
            "fastsam-box": (SEGMENT_REGION,),
            "fastsam-part": (SEGMENT_REGION, SEGMENT_PART),
        }.get(self.options.backend, (SEGMENT,))
        self.actions(ActionExecutor(
            self.journal, state=SegmentationState(),
            skills=tuple(skill.bind(self.segment) for skill in skills),
            backend=self.options.backend, features_extra={"robot_controls": False,
                                          "controller_flush": False,
                                          "verification": "task_scoped_segmentation",
                                          "stop_scope": "segmentation_result"},
        ))
        self.transport.register_handler("request/segmentation/data", self.read, binary=True)
        self.transport.register_handler("request/segmentation/info", self.info_result)
        self.transport.register_handler("request/segmentation/latest", self.latest)
        self.discover()
        self.spawn(self._expire())

    async def _close_worker(self):
        await asyncio.to_thread(self.worker.shutdown, wait=True, cancel_futures=True)

    async def _expire(self):
        while True:
            await asyncio.sleep(1)
            self.cache.expire()

    async def _ground_part(self, args, location, rgb, observed_at, stop):
        from PIL import Image

        from wrs_agent.planner.grasp_part_prompts import check_box

        # Keep the original camera age and reserve time for FastSAM and the next skill.
        budget = self.cache.ttl - (time.monotonic() - observed_at) - 5
        if budget <= 0:
            raise AgentError("observation_stale")
        object_box = location.object.box
        ground = asyncio.create_task(self.part_client.ground(
            args.goal, location.target, args.part, Image.fromarray(rgb), object_box=object_box,
        ))
        stopped = asyncio.create_task(stop.wait())
        try:
            done, _ = await asyncio.wait((ground, stopped), timeout=budget,
                                         return_when=asyncio.FIRST_COMPLETED)
            if stopped in done or stop.is_set():
                return None
            if ground not in done:
                raise AgentError("observation_stale")
            answer = ground.result()
        except ValidationError:
            raise AgentError("part_grounding_failed") from None
        finally:
            for task in (ground, stopped):
                if not task.done():
                    task.cancel()
            await asyncio.gather(ground, stopped, return_exceptions=True)
        info = location.observation
        box = answer.pixel_box(info.width, info.height)
        if box is None:
            raise AgentError("part_not_found")
        if reason := check_box(box, rgb.shape):
            raise AgentError(reason)
        if not part_box_overlaps_object(box, object_box):
            raise AgentError("part_box_outside_object")
        return PartProvenance(
            name=args.part, box=box, grounding_model=self.part_client.config.model,
        )

    async def segment(self, state, args, stop, progress):
        self.latest_request = None
        owner = artifact_owner(self.executor)
        source = (self.options.locator if self.options.backend in BOX_BACKENDS
                  else self.options.camera)
        try:
            location = None
            part = None
            if self.options.backend in BOX_BACKENDS:
                await self.registry.refresh([self.options.locator])
                located, age = await read_location(self.connect(self.options.locator),
                                                   args.location_id, owner.task_id)
                observed_at = time.monotonic() - age
                location = located.info
                info, arrays, target = location.observation, located.arrays, location.target
                if isinstance(args, PartRegionArgs):
                    if stop.is_set():
                        return False
                    part = await self._ground_part(args, location, arrays["rgb"], observed_at, stop)
                    if part is None:
                        return False
                    inference = partial(self.segmenter.segment_box_local_retry, arrays["rgb"],
                                        part.box)
                else:
                    inference = partial(self.segmenter.segment_box, arrays["rgb"],
                                        location.object.box)
            else:
                await self.registry.refresh([self.options.camera])
                info, arrays, age = await CameraClient(self.connect(self.options.camera)).read()
                observed_at = time.monotonic() - age
                if age > 2:
                    raise SkillFailure("camera_stale")
                target = args.target
                inference = partial(self.segmenter.segment, arrays["rgb"], target, args.selection)
            if stop.is_set():
                return False
            mask, score = await self.loop.run_in_executor(self.worker, inference)
            if stop.is_set():
                return False
            data = SegmentData(SegmentInfo(
                segment_id=new_id(), owner=owner, observation=info,
                target=target, location=location, part=part,
                model=self.segmenter.name, weights_sha256=self.segmenter.sha256,
                score_kind=self.segmenter.score_kind,
                text_weights_sha256=getattr(self.segmenter, "text_sha256", None),
                score=score, pixel_count=int(mask.sum()),
            ), arrays, mask)
            self.cache.put(data.info.segment_id, owner, data, nbytes=data.nbytes,
                           observed_at=observed_at)
        except AgentError as exc:
            # Keep discovery/ownership context; never stringify structured errors.
            if exc.code == "request_timeout":
                raise SkillFailure(error_info(
                    "segmentation_source_timeout", node_id=exc.error.node_id or source,
                    stage=exc.error.stage,
                )) from exc
            raise SkillFailure(exc.error) from exc
        except TimeoutError as exc:
            raise SkillFailure(error_info("segmentation_source_timeout", node_id=source)) from exc
        except ValueError as exc:
            raise SkillFailure(known_error_info(str(exc)) or
                               error_info("invalid_segmentation_output")) from exc
        except RuntimeError as exc:
            raise SkillFailure(error_info("segmentation_inference_failed")) from exc
        state.completed += 1
        self.latest_request = ArtifactRequest(artifact_id=data.info.segment_id,
                                              task_id=owner.task_id)
        return SegmentOutput(segment_id=data.info.segment_id, observation_id=info.observation_id)

    async def latest(self, payload):
        Empty.model_validate(payload)
        request = self.latest_request
        if request is None:
            raise AgentError("segment_not_available")
        data, age = self.cache.get(request, self.executor)
        return LatestSegment(segment_id=data.info.segment_id, task_id=data.info.owner.task_id,
                             observation_id=data.info.observation.observation_id,
                             age_s=age).model_dump()

    async def read(self, payload):
        request = ArtifactRequest.model_validate(payload)
        data, age = self.cache.get(request, self.executor)
        packet = await asyncio.to_thread(pack_segment, data, age)
        # Stop may have arrived while serializing the image; do not reply with revoked data.
        self.cache.get(request, self.executor)
        return packet

    async def info_result(self, payload):
        data, age = self.cache.get(ArtifactRequest.model_validate(payload), self.executor)
        return {"segment": data.info.model_dump(), "age_s": age}


# Keep existing SAM3 launch scripts working; there is still only one node implementation.
Sam3Node = SegmentationNode
Sam3Options = SegmentationOptions
