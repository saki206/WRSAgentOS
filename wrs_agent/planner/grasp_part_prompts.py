"""抓取部位的通用提示词、结构化输出和坐标校验。"""

import re
from typing import Literal

from pydantic import Field, model_validator

from wrs_agent.schemas import Boundary

# 第一阶段只输入口令。模型按任务和部件的物理功能提出首选部位，
# 是否存在、是否承重和是否适配夹爪，都需要后续确认。
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

# 使用短英文名称，方便把目标和部位传给后续视觉模块。
PART_NAME = re.compile(r"[a-z][a-z0-9 _-]*\Z")


# 发给模型的 JSON Schema 只包含需要生成的字段。
# 本地模型另行标记待视觉确认，并限制名称格式和字段长度。
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
    """一个首选部位，以及推荐理由和采用前需要核实的条件。"""

    part: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=1, max_length=500)
    condition: str = Field(min_length=1, max_length=500)


class GraspPartSuggestion(Boundary):
    """文字阶段的输出，供视觉定位和原仓库目标定位技能使用。"""

    object: str = Field(min_length=1, max_length=64)
    action: str = Field(min_length=1, max_length=80)
    preferred_parts: list[PreferredPart] = Field(min_length=1, max_length=1)
    avoid_parts: list[str] = Field(default_factory=list, max_length=5)
    needs_visual_confirmation: Literal[True] = True

    @model_validator(mode="after")
    def check_suggestion(self):
        # 除了接口要求的字符串类型，还要校验物体名和部位名的实际格式。
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
    """将用户口令、目标物体和首选部位填入第二阶段提示词。"""
    # 核实部位是否属于当前实例。固定附件可能通过中间结构连接，
    # 独立支架、其他物体和背景不应被框入。
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
    # 要求模型在证据不足时返回空框，便于程序中止本次分割。
    return prompt + (
        "Return only JSON with fields box, status, and evidence. box is four integers or null. "
        "Use matched exactly when box is not null. Other statuses are target_not_visible, "
        "part_not_visible, ambiguous_target, ambiguous_part, and uncertain_attachment. "
        "Give one short sentence of visible evidence or uncertainty."
    )


class PartBox(Boundary):
    """视觉定位输出。坐标使用 0–999 的归一化范围。"""

    box: list[int] | None = Field(min_length=4, max_length=4)
    status: Literal[
        "matched", "target_not_visible", "part_not_visible", "ambiguous_target",
        "ambiguous_part", "uncertain_attachment",
    ]
    evidence: str = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def valid_box(self):
        # 状态必须与框是否存在一致，有框时再检查坐标范围和顺序。
        if (self.box is not None) != (self.status == "matched"):
            raise ValueError("part_box_status_mismatch")
        if self.box is not None:
            x1, y1, x2, y2 = self.box
            if not (0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999):
                raise ValueError("invalid_part_box")
        return self

    def pixel_box(self, width: int, height: int) -> list[float] | None:
        """按原图尺寸换算像素边界，右侧和下侧使用不含端点的坐标。"""
        if self.box is None:
            return None
        return [value * size / 999 for value, size in zip(
            self.box, (width, height, width, height), strict=True,
        )]


def check_box(box, shape, minimum=2, max_fraction=1):
    """检查像素宽高和占图比例，返回失败原因或 None。"""
    x1, y1, x2, y2 = box
    if min(x2 - x1, y2 - y1) < minimum:
        return "part_box_too_small"
    if (x2 - x1) * (y2 - y1) > max_fraction * shape[0] * shape[1]:
        return "part_box_area_exceeded"
    return None
