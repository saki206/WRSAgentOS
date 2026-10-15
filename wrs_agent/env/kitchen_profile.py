"""Calibrated object affordances, separate from robot skill contracts and object IDs."""
import tomllib
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field
from scipy.spatial.transform import Rotation

from wrs_agent.executor import SkillFailure
from wrs_agent.schemas import Boundary, Vector3


class ObjectProfile(Boundary):
    asset: str
    kind: Literal["food", "strainer", "vessel", "surface"]
    grasp_pos: Vector3 | None = None
    grasp_rpy: Vector3 | None = None
    opening: float | None = Field(default=None, gt=0, le=0.086)
    approach_opening: float | None = Field(default=None, gt=0, le=0.086)
    upright_rpy: Vector3 | None = None
    bottom: Vector3 = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    rim: Vector3 | None = None
    opening_axis: Vector3 = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    radius: float = Field(gt=0, le=1)
    half_height: float = Field(default=0.0, ge=0)
    working_offset: Vector3 | None = None
    contact_axis: int | None = Field(default=None, ge=0, le=2)
    contact_bounds: list[float] | None = Field(default=None, min_length=2, max_length=2)
    contact_x_bounds: list[float] | None = Field(default=None, min_length=2, max_length=2)


def matrix(pos, rpy=None):
    tf = np.eye(4)
    tf[:3, 3] = pos
    if rpy is not None:
        tf[:3, :3] = Rotation.from_euler("xyz", rpy, degrees=True).as_matrix()
    return tf


def match_profiles(objects):
    path = Path(__file__).parents[1] / "assets/kitchen/actions.toml"
    profiles = tomllib.loads(path.read_text(encoding="utf-8"))["profiles"]
    by_asset = {entry["asset"]: ObjectProfile.model_validate(entry) for entry in profiles.values()}
    return {name: by_asset[obj.geometry.asset] for name, obj in objects.items()
            if obj.geometry is not None and obj.geometry.kind == "mesh"
            and obj.geometry.asset in by_asset}


def role_ids(objects):
    roles = {}
    for name, profile in match_profiles(objects).items():
        if profile.kind in roles:
            raise SkillFailure("kitchen_role_ambiguous")
        roles[profile.kind] = name
    if not {"food", "strainer", "vessel", "surface"} <= roles.keys():
        raise SkillFailure("kitchen_recipe_objects_missing")
    return roles
