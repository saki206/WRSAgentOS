"""Route voice goals to the shared grasp-preparation plan."""

import re
import unicodedata

from pydantic import ValidationError

from wrs_agent.errors import AgentError
from wrs_agent.planner import PlanDecision
from wrs_agent.schemas import Plan
from wrs_agent.system import step

PART_GRASP_SKILLS = {"locate_object", "segment_part", "infer_grasps"}

# Only the action wording is used here. The model still resolves the target,
# its requested attributes and whether the complete request is supported.
ZH_REQUEST_PREFIX = r"(?:(?:请你?|帮我|麻烦你?|给我|先)[\s,]*)*"
GRASP_CUE = re.compile(
    r"(?:抓|夹)(?:起|住|稳|紧|一下)|(?:拿|捡|拾|提|拎)起"
    rf"|^{ZH_REQUEST_PREFIX}(?:抓|夹)(?!爪)[^\W_]"
    r"|(?:把|将)[^,;.!?。]{1,80}(?:抓取|夹取)(?:[。.!?,;]|$)"
    r"|(?:生成|计算|预测|规划|给出)[^,;.!?。]{0,80}(?:抓姿|抓取候选)"
    r"|\b(?:grasp|grab|grip|lift)\b|\bpick\s+[^.!?。！？]{0,80}\bup\b",
    re.IGNORECASE,
)
ORDINARY_GOAL = re.compile(
    rf"^{ZH_REQUEST_PREFIX}"
    r"(?:向[上下左右前后]移动|移动|回到|旋转|转动|打开|张开|关闭|闭合|夹紧夹爪"
    r"|查看|显示|查询|怎么|如何)"
    r"|^(?:please\s+)?(?:move|rotate|turn|open|close|show|display|review|how)\b"
    r"|^(?:please\s+)?lift\s+(?:(?:the|your|robot(?:'s)?)\s+)*(?:arm|gripper)\b",
    re.IGNORECASE,
)


def grasp_preparation_plan(target, *, selection="only", goal="", part=None):
    """Build the same task and artifact bindings for the voice and grasping examples."""
    locate = step("locate_object", target=target, selection=selection)
    location = {"location_id": {"step_id": locate.step_id, "field": "location_id"}}
    if part is None:
        segment = step("segment_region", after=locate, inputs=location)
    else:
        segment = step("segment_part", after=locate, inputs=location, goal=goal, part=part)
    propose = step("infer_grasps", after=segment,
                   inputs={"segment_id": {"step_id": segment.step_id, "field": "segment_id"}})
    return Plan(steps=[locate, segment, propose])


class VoiceGraspPlanner:
    """Keep plan construction local and delegate unrelated goals to the existing planner."""

    def __init__(self, part_client_factory, planner):
        self.part_client_factory = part_client_factory
        self.planner = planner
        self.last_usage = {}

    async def plan(self, request):
        self.last_usage = {}
        goal = unicodedata.normalize("NFKC", request.user_goal).strip()
        if ORDINARY_GOAL.match(goal) or not GRASP_CUE.search(goal):
            return await self._plan_ordinary_goal(request)
        try:
            async with self.part_client_factory() as client:
                routed = await client.suggest_voice_goal(request.user_goal)
        except ValidationError:
            raise AgentError("part_suggestion_invalid", stage="planning") from None
        if routed.intent == "other":
            return await self._plan_ordinary_goal(request)
        if routed.intent == "clarify":
            return PlanDecision(kind="clarify", text="请单独说出要夹起的一个物体。")
        if not PART_GRASP_SKILLS.issubset(skill["name"] for skill in request.skills):
            raise AgentError("part_grasp_skills_unavailable", stage="planning")
        suggestion = routed.suggestion
        plan = grasp_preparation_plan(suggestion.object, selection=routed.selection,
                                      goal=request.user_goal,
                                      part=suggestion.preferred_parts[0].part)
        return PlanDecision(kind="execute", plan=plan)

    async def _plan_ordinary_goal(self, request):
        decision = await self.planner.plan(request)
        self.last_usage = getattr(self.planner, "last_usage", {})
        return decision
