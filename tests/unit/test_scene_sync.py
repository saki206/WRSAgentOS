from types import SimpleNamespace

import pytest

from wrs_agent.env.wrs import sync_scene_objects
from wrs_agent.schemas import ObjectData

IDENTITY = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]


class RenderedBody:
    def __init__(self, **values):
        self.__dict__.update(values)
        self.pose_updates = 0

    def set_pos_rotmat(self, *, pos, rotmat):
        self.pos, self.rotmat = pos, rotmat
        self.pose_updates += 1

    def add_to_scene(self, scene):
        scene.added.append(self)

    def remove_from_scene(self, scene):
        scene.removed.append(self)


@pytest.fixture
def render_state():
    created = []

    def box(**values):
        body = RenderedBody(**values)
        created.append(body)
        return body

    wrs = SimpleNamespace(wssop=SimpleNamespace(box=box))
    scene = SimpleNamespace(added=[], removed=[])
    return wrs, scene, {}, created


def observed_body(**updates):
    return ObjectData.model_validate({
        "source": "simulation", "observed_at_ns": 1,
        "pos": [0., 0., 0.], "rotmat": IDENTITY,
        "geometry": {"kind": "box", "xyz_lengths": [.1, .1, .1]},
        **updates,
    })


def test_moving_object_keeps_the_same_rendered_body_for_every_snapshot(render_state):
    wrs, scene, displayed, created = render_state
    for index in range(100):
        data = observed_body(pos=[index*.001, 0., 0.], observed_at_ns=index+1,
                             location="gripper")
        sync_scene_objects(wrs, scene, {"tool": data}, displayed)
        assert displayed["tool"][1] is created[0]
        assert displayed["tool"][1].pos == data.pos
    assert len(created) == len(scene.added) == 1
    assert scene.removed == []
    assert created[0].pose_updates == 99


def test_metadata_only_refresh_does_not_reload_geometry_or_change_pose(render_state):
    wrs, scene, displayed, created = render_state
    sync_scene_objects(wrs, scene, {"tool": observed_body()}, displayed)
    update = observed_body(observed_at_ns=2, source="camera", location="gripper", label="工具")
    sync_scene_objects(wrs, scene, {"tool": update}, displayed)
    assert displayed["tool"][0] == update
    assert len(created) == 1 and scene.removed == []
    assert created[0].pose_updates == 0
    update.pos[0] = 42.
    assert displayed["tool"][0].pos[0] == 0.


def test_color_and_opacity_change_in_place(render_state):
    wrs, scene, displayed, created = render_state
    sync_scene_objects(wrs, scene, {"tool": observed_body()}, displayed)
    sync_scene_objects(wrs, scene, {"tool": observed_body(rgb=[1., 0., 0.], alpha=.4)}, displayed)
    assert len(created) == 1 and scene.removed == []
    assert created[0].rgb == [1., 0., 0.] and created[0].alpha == .4


def test_changed_geometry_replaces_the_old_body(render_state):
    wrs, scene, displayed, created = render_state
    sync_scene_objects(wrs, scene, {"tool": observed_body()}, displayed)
    resized = observed_body(geometry={"kind": "box", "xyz_lengths": [.2, .1, .1]})
    sync_scene_objects(wrs, scene, {"tool": resized}, displayed)
    assert len(created) == 2 and scene.removed == [created[0]]
    assert displayed["tool"][1] is created[1]


@pytest.mark.parametrize("unknown", [{"valid": False}, {"pos": None},
                                     {"rotmat": None}, {"geometry": None}])
def test_unavailable_geometry_or_pose_removes_the_displayed_body(render_state, unknown):
    wrs, scene, displayed, created = render_state
    sync_scene_objects(wrs, scene, {"tool": observed_body()}, displayed)
    sync_scene_objects(wrs, scene, {"tool": observed_body(**unknown)}, displayed)
    assert displayed == {} and scene.removed == [created[0]]
    sync_scene_objects(wrs, scene, {"tool": observed_body()}, displayed)
    assert len(created) == 2 and displayed["tool"][1] is created[1]


def test_removed_object_leaves_no_rendered_body(render_state):
    wrs, scene, displayed, created = render_state
    sync_scene_objects(wrs, scene, {"tool": observed_body()}, displayed)
    sync_scene_objects(wrs, scene, {}, displayed)
    assert displayed == {} and scene.removed == [created[0]]
