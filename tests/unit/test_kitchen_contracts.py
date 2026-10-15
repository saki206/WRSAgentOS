import math

import pytest
from pydantic import ValidationError

from wrs_agent.env.kitchen_profile import role_ids
from wrs_agent.env.wrs import ROOT
from wrs_agent.recipes.noodles import noodle_steps
from wrs_agent.scene import load_scene
from wrs_agent.schemas import Plan
from wrs_agent.skills.robot import SKILLS, HoldInArgs, PickArgs, PlaceOnArgs


def test_recipe_is_nine_existing_atomic_contracts_with_runtime_dependencies():
    objects = load_scene(ROOT / "wrs_agent/assets/kitchen/scene.toml")
    steps = noodle_steps(objects)
    Plan(steps=steps)
    assert [s.skill for s in steps] == [
        "pick", "reorient", "place_on", "pick", "seat_in", "pick", "hold_in", "pour", "place_on",
    ]
    for i, item in enumerate(steps):
        SKILLS[item.skill].arguments.model_validate(item.args)
        assert item.depends_on == ([steps[i-1].step_id] if i else [])
    assert steps[6].args["seconds"] == 10.0
    assert steps[-1].args["placement"] == "initial"
    assert "cook_noodles" not in SKILLS


def test_profiles_and_recipe_follow_observed_ids_not_fixed_noodle_names():
    objects = load_scene(ROOT / "wrs_agent/assets/kitchen/scene.toml")
    renamed = {"observed_"+name: obj for name, obj in objects.items()}
    roles = role_ids(renamed)
    assert roles["strainer"] == "observed_skimmer"
    steps = noodle_steps(renamed)
    assert steps[0].args["object"] == roles["strainer"]
    assert steps[4].args["object"] == roles["food"]
    assert steps[6].args["container"] == roles["vessel"]
    assert steps[7].args["target"] == roles["surface"]


@pytest.mark.parametrize("args", [
    {"object": "skimmer", "container": "skimmer"},
    {"object": "skimmer", "container": "pot", "seconds": math.nan},
    {"object": "skimmer", "container": "pot", "seconds": 121.0},
    {"object": "skimmer", "container": "pot", "motion_speed": .1},
])
def test_hold_contract_rejects_invalid_combinations(args):
    with pytest.raises(ValidationError):
        HoldInArgs.model_validate(args)


def test_existing_pick_and_default_placement_keep_their_original_arguments():
    assert PickArgs(object="A").motion_speed == 1.0
    assert PlaceOnArgs(object="A", surface="plate").placement == "center"
    assert HoldInArgs(object="skimmer", container="pot").seconds == 10.0
    assert "release" in SKILLS["seat_in"].description
    assert "No release" in SKILLS["hold_in"].description


def test_ambiguous_affordances_do_not_silently_choose_a_target():
    objects = load_scene(ROOT / "wrs_agent/assets/kitchen/scene.toml")
    objects["other_plate"] = objects["plate"]
    with pytest.raises(Exception, match="kitchen_role_ambiguous"):
        noodle_steps(objects)
