from types import SimpleNamespace

from wrs_agent.nodes.ui import scene_view


async def test_viewer_keeps_its_initial_snapshot_when_polling_drops_during_model_load(monkeypatch):
    snapshot = SimpleNamespace(data=SimpleNamespace(facts={"robot_model": "fafu_baseV1"}))
    dashboard = SimpleNamespace(robot=snapshot, viewer={})
    view = scene_view.SceneView(dashboard, 8004)
    wrs = object()
    received = []

    def import_model():
        dashboard.robot = None
        return wrs

    def build_model(runtime, initial):
        received.append((runtime, initial))
        raise RuntimeError("test_completed_initial_build")

    monkeypatch.setattr(scene_view, "_port_in_use", lambda port: False)
    monkeypatch.setattr(scene_view, "_bypass_proxy_for_loopback", lambda: None)
    monkeypatch.setattr(scene_view, "load_wrs", import_model)
    monkeypatch.setattr(view, "_build", build_model)
    await view.run()
    assert received == [(wrs, snapshot)]
    assert "test_completed_initial_build" in dashboard.viewer["detail"]
