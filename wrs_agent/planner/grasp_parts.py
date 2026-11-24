"""Route voice goals to the shared grasp-preparation plan."""

from pydantic import ValidationError

from wrs_agent.errors import AgentError
from wrs_agent.planner import PlanDecision
from wrs_agent.schemas import Plan
from wrs_agent.system import step

PART_GRASP_SKILLS = {"locate_object", "segment_part", "infer_grasps"}


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

    def __init__(self, part_client, planner):
        self.part_client = part_client
        self.planner = planner
        self.last_usage = {}

    async def plan(self, request):
        self.last_usage = {}
        try:
            routed = await self.part_client.suggest_voice_goal(request.user_goal)
        except ValidationError:
            raise AgentError("part_suggestion_invalid", stage="planning") from None
        if routed.intent == "other":
            decision = await self.planner.plan(request)
            self.last_usage = getattr(self.planner, "last_usage", {})
            return decision
        if routed.intent == "clarify":
            return PlanDecision(kind="clarify", text="请单独说出要夹起的一个物体。")
        if not PART_GRASP_SKILLS.issubset(skill["name"] for skill in request.skills):
            raise AgentError("part_grasp_skills_unavailable", stage="planning")
        suggestion = routed.suggestion
        plan = grasp_preparation_plan(suggestion.object, selection=routed.selection,
                                      goal=request.user_goal,
                                      part=suggestion.preferred_parts[0].part)
        return PlanDecision(kind="execute", plan=plan)
