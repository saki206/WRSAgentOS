"""A nine-step plan, not an additional robot skill or nested action executor."""
from typing import Literal

from pydantic import Field

from wrs_agent import step
from wrs_agent.env.kitchen_profile import role_ids
from wrs_agent.schemas import Boundary

RECIPE_TIMEOUT = 600.0


class NoodleOptions(Boundary):
    seconds: float = Field(default=10.0, ge=0.1, le=120.0)
    motion_speed: float = Field(default=1.0, ge=0.25, le=3.0)
    group: Literal["all", "load", "cook", "serve"] = "all"


def noodle_steps(objects, *, seconds=10.0, motion_speed=1.0, group="all"):
    options = NoodleOptions(seconds=seconds, motion_speed=motion_speed, group=group)
    roles = role_ids(objects)
    food, net, pot, plate = (roles[kind] for kind in ("food", "strainer", "vessel", "surface"))
    tables = [name for name, obj in objects.items()
              if obj.geometry is not None and obj.geometry.kind == "box"
              and obj.geometry.xyz_lengths[0] > .5 and obj.geometry.xyz_lengths[1] > .5]
    if len(tables) != 1:
        raise ValueError("recipe_support_surface_missing_or_ambiguous")
    table, = tables
    groups = {
        "load": [("pick", {"object": net}),
                 ("reorient", {"object": net, "orientation": "upright"}),
                 ("place_on", {"object": net, "surface": table, "placement": "working"}),
                 ("pick", {"object": food}),
                 ("seat_in", {"object": food, "container": net})],
        "cook": [("pick", {"object": net}),
                 ("hold_in", {"object": net, "container": pot, "seconds": options.seconds})],
        "serve": [("pour", {"source": net, "target": plate}),
                  ("place_on", {"object": net, "surface": table, "placement": "initial"})],
    }
    selected = ["load", "cook", "serve"] if group == "all" else [group]
    result = []
    for part in selected:
        for skill, args in groups[part]:
            result.append(step(skill, **args, motion_speed=options.motion_speed,
                               after=result[-1] if result else ()))
    return result
