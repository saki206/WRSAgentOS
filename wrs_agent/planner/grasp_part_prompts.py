"""Shared prompts and output contracts for language-guided part segmentation."""

import re
from typing import Literal

from pydantic import Field, model_validator

from wrs_agent.schemas import Boundary

SYSTEM_PROMPT = (
    "You suggest where the robot itself should grasp an object with a parallel-jaw gripper. "
    "Optimize for the robot's own stable lifting, carrying, or use of the object. "
    "Do not optimize for handing an object to another person or leaving its handle free. "
    "Choose by physical function, not a fixed mapping from object category to part. "
    "A sturdy ergonomic handle is usually first when it exists and fits the gripper. "
    "A category name alone does not establish that this instance has a handle. "
    "If the user says an object has no handle, never propose one. "
    "Prefer load-bearing parts over thin, fragile, detachable, or functional surfaces. "
    "For sharp or hot tools, never propose a blade or hot surface as a fallback grip. "
    "If the instruction mentions heat, sharpness, chemical exposure, dirt, or food contact, "
    "record relevant hazards and protect functional surfaces. Treat unknown temperature, "
    "contamination, and gripping clearance as conditions to check when relevant. "
    "Infer the intended action from the user's words. "
    "Select exactly one best physical part. Do not propose fallback or alternative parts. "
    "Use lowercase English ASCII physical-part names such as 'handle', 'side panel', or 'body'; "
    "do not put Chinese text or parenthetical translations in part names. "
    "Set object to a short lowercase English object noun phrase, without the part name or "
    "scene description, suitable for visual object detection. "
    "Explicit user part preferences take priority. If an object has no known distinct part, use "
    "'whole object' or 'body'. Do not assume a possible part is visible or present. "
    "Describe the part by its physical role. Do not invent color, material, shape, dimensions, "
    "or scene position unless explicitly supplied by the user. "
    "Treat the selected part as a hypothesis for visual verification, not an observed fact. "
    "Put presence, gripper fit, and task-dependent requirements in each condition. "
    "The object's dimensions and gripper opening are unknown: never assert that a part fits, "
    "or state a numerical or directional size relationship; make fit a condition to verify. "
    "Avoid-part names must belong to the target object, not to other scene objects. "
    "Do not output coordinates, grasp poses, motor commands, or claims of visual confirmation."
)

PART_NAME = re.compile(r"[a-z][a-z0-9 _-]*\Z")


PART_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "object": {"type": "string"},
        "action": {"type": "string"},
        "preferred_parts": {
            "type": "array",
            "minItems": 1,
            "maxItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "part": {"type": "string"},
                    "reason": {"type": "string"},
                    "condition": {"type": "string"},
                },
                "required": ["part", "reason", "condition"],
            },
        },
        "avoid_parts": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["object", "action", "preferred_parts", "avoid_parts"],
}


class PreferredPart(Boundary):
    part: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=1, max_length=500)
    condition: str = Field(min_length=1, max_length=500)


class GraspPartSuggestion(Boundary):
    object: str = Field(min_length=1, max_length=80)
    action: str = Field(min_length=1, max_length=80)
    preferred_parts: list[PreferredPart] = Field(min_length=1, max_length=1)
    avoid_parts: list[str] = Field(default_factory=list, max_length=5)
    needs_visual_confirmation: Literal[True] = True

    @model_validator(mode="after")
    def check_suggestion(self):
        if not PART_NAME.fullmatch(self.object):
            raise ValueError("invalid_object_name")
        names = {item.part.casefold() for item in self.preferred_parts}
        if len(names) != len(self.preferred_parts):
            raise ValueError("duplicate_grasp_part")
        if any(not PART_NAME.fullmatch(item.part) for item in self.preferred_parts):
            raise ValueError("invalid_preferred_part_name")
        if any(not PART_NAME.fullmatch(item) or len(item) > 64 for item in self.avoid_parts):
            raise ValueError("invalid_avoid_part")
        return self


def grounding_prompt(goal: str, object_description: str, part: str) -> str:
    """Shared, category-independent prompt for part localization model comparisons."""
    prompt = (
        "Locate a proposed grasp part on the target object in this image. "
        f"User command: {goal}\n"
        f"Target object: {object_description}. Proposed part: {part}. "
        "First identify the target instance as a physical assembly, including its fixed "
        "attachments and accessories that would normally move with it during carrying. "
        "A part may connect through a fixed frame, bracket, or intermediate connector; "
        "direct contact with the main object's surface is not required. Use visible mounting "
        "and structural continuity as evidence; observing actual movement is not required. "
        "Distinguish such attached components from a separate external stand or support on "
        "which the object merely rests. Do not exclude a part just because it attaches to "
        "a fixed carrying frame rather than the main surface. "
        "Then verify that the proposed part is visibly attached to and belongs to "
        "that same instance. "
        "The part proposal is a hypothesis, "
        "not proof that the part exists. Use the actual visible structure, including perspective "
        "and partial occlusion; do not impose a typical category shape "
        "or an assumed color or material. "
        "Identify the part by its load-bearing function and physical connection to the object. "
        "Do not select another object's support or stand, an unrelated fastener or decoration, "
        "or a background region merely because it resembles the proposed part. Judge ownership "
        "and function relative to this target; no object category is inherently excluded. "
        "Do not silently substitute a different part. "
        "If there are multiple plausible target instances or part matches and the request does not "
        "disambiguate them, return null. Also return null if attachment or part identity cannot be "
        "established from the visible evidence. "
        "For one verified match, return a tight axis-aligned box enclosing only the visible target "
        "part, not the whole object or its surroundings. Use normalized image coordinates "
        "[x1,y1,x2,y2], integers from 0 to 999. "
    )
    return prompt + (
        "Return only JSON with fields box, status, and evidence. box is four integers or null. "
        "Use matched exactly when box is not null. Other statuses are target_not_visible, "
        "part_not_visible, ambiguous_target, ambiguous_part, and uncertain_attachment. "
        "Give one short sentence of visible evidence or uncertainty."
    )


class PartBox(Boundary):
    box: list[int] | None = Field(min_length=4, max_length=4)
    status: Literal[
        "matched", "target_not_visible", "part_not_visible", "ambiguous_target",
        "ambiguous_part", "uncertain_attachment",
    ]
    evidence: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def valid_box(self):
        if (self.box is not None) != (self.status == "matched"):
            raise ValueError("part_box_status_mismatch")
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            if not (0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999):
                raise ValueError("invalid_part_box")
        return self

    def pixel_box(self, width: int, height: int) -> list[float] | None:
        if self.box is None:
            return None
        return [value * size / 999 for value, size in zip(
            self.box, (width, height, width, height), strict=True,
        )]
