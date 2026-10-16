"""Optional WRS 3D page drawn from the robot node's snapshots; it never sends motion."""

import asyncio
import os
import socket
import threading

from wrs_agent.env.wrs import ROOT, display_model, load_wrs, serve_viewer_hub, sync_scene_objects


def _port_in_use(port):
    with socket.socket() as probe:
        probe.settimeout(0.25)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _bypass_proxy_for_loopback():
    # websockets honours HTTP(S)_PROXY; a system proxy would swallow ws://127.0.0.1/publish.
    for name in ("NO_PROXY", "no_proxy"):
        hosts = [item for item in os.environ.get(name, "").split(",") if item]
        os.environ[name] = ",".join(dict.fromkeys([*hosts, "127.0.0.1", "localhost"]))


class SceneView:
    def __init__(self, dashboard, port):
        self.dashboard, self.port = dashboard, port
        self.world = None
        self._thread = None
        self.controls = None

    def _status(self, state, detail):
        self.dashboard.viewer = {"state": state, "detail": detail}

    async def run(self):
        url = f"http://127.0.0.1:{self.port}/"
        if _port_in_use(self.port):
            self._status("external", f"{self.port} 端口已有画面服务，直接嵌入它。")
            return
        self._status("starting", "正在加载 WRS 模型…")
        _bypass_proxy_for_loopback()
        try:
            for _ in range(150):
                if self.dashboard.robot is not None:
                    break
                await asyncio.sleep(.1)
            if self.dashboard.robot is None:
                raise RuntimeError("robot_snapshot_not_ready")
            initial = self.dashboard.robot
            wrs = await asyncio.to_thread(load_wrs)
            await asyncio.to_thread(self._build, wrs, initial)
        except Exception as exc:
            self._status("failed", f"WRS 画面不可用：{type(exc).__name__}: {exc}")
            return
        hub = asyncio.create_task(serve_viewer_hub(self.port))
        try:
            for _ in range(50):
                if hub.done() or _port_in_use(self.port):
                    break
                await asyncio.sleep(0.1)
            if hub.done():
                await hub
                raise RuntimeError("viewer_hub_stopped")
            self._thread = threading.Thread(
                target=self.world.run, name="wrs-ui-viewer", daemon=True,
            )
            self._thread.start()
            self._status("ready", url)
            await hub
            self._status("failed", "WRS 画面服务已停止。")
        except Exception as exc:
            self._status("failed", f"WRS 画面服务失败：{type(exc).__name__}: {exc}")
        finally:
            self.world.close()
            hub.cancel()
            await asyncio.gather(hub, return_exceptions=True)

    def _build(self, wrs, initial):
        world = wrs.wvw.World(
            cam_pos=(1.65, -1.85, 1.55) if initial.data.facts.get("robot_model")
            == "fafu_baseV1" else (1.8, 1.4, 1.2), cam_lookat_pos=(0.3, 0, 0.35),
            port=self.port, hz=10, auto_start_hub=False,
        )
        world.set_caption("WRS-Agent · UI 节点")
        model = display_model(initial)
        model.robot.add_to_scene(world.scene)
        if initial.data.facts.get("manipulation_profile"):
            from wrs_agent.env.recipe_viewer import CookingControls

            self.controls = CookingControls(world, self.dashboard.info["endpoint"],
                                            self.dashboard.info["env_id"],
                                            ROOT / "examples/wrs/bindings.toml")
        wrs.wssop.frame().add_to_scene(world.scene)
        displayed = {}
        board = self.dashboard

        def update(dt):
            # Display only: the latest snapshot comes from the dashboard's own polling.
            snapshot = board.robot
            if snapshot is None or snapshot.data.kind != "scene":
                return
            try:
                state = snapshot.data.robot.kinematics
                if state is not None and state.valid:
                    model.sync(state.qs, state.gripper_width)
                sync_scene_objects(wrs, world.scene, snapshot.data.objects, displayed)
                if self.controls is not None:
                    self.controls.update(snapshot)
            except Exception as exc:
                self._status("failed", f"画面刷新失败：{type(exc).__name__}: {exc}")

        world.schedule_interval(update, 0.1)
        self.world = world

    async def close(self):
        if self.world is not None:
            self.world.close()
        if self._thread is not None:
            await asyncio.to_thread(self._thread.join, 2)
        if self.controls is not None:
            await asyncio.to_thread(self.controls.close)
