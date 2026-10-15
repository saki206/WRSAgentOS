"""Cooking controls use an independent asynchronous client; drawing never waits for a task."""

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor

from wrs_agent import ActionState, System
from wrs_agent.env.kitchen_profile import role_ids
from wrs_agent.recipes.noodles import RECIPE_TIMEOUT, noodle_steps
from wrs_agent.schemas import ControlRequest, new_id


class CookingControls:
    def __init__(self, world, endpoint, env_id, bindings):
        self.world = world
        self.endpoint, self.env_id, self.bindings = endpoint, env_id, bindings
        self.cook_seconds, self.motion_speed = 10.0, 1.0
        self.ready = Future()
        self.job = None
        self.active_action = None
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cooking-ui")
        self.worker = self.pool.submit(asyncio.run, self._connect())
        try:
            self.ready.result(timeout=15)
        except Exception:
            self.pool.shutdown(wait=True)
            raise
        self.panel = world.ui.add_panel(
            "cooking", title="基础技能组合 · 煮面",
            description="抓取 → 扶正 → 入网 → 持网计时 → 倾倒",
            width=292,
        )
        self.panel.add_label("stage", label="当前动作", value="准备夹面")
        self.panel.add_label("food", label="面的位置", value="盘中")
        self.panel.add_label("gap", label="夹爪间距", value="86.0 毫米")
        self.panel.add_label("timer", label="煮面计时", value="0 / 10 秒")
        self.panel.add_label("result", label="操作结果", value="等待开始")
        self.panel.add_slider(
            "seconds", label="煮面等待", min_value=1, max_value=120, step=1, value=10, unit="秒",
            on_change=lambda value: setattr(self, "cook_seconds", value),
        )
        self.panel.add_slider(
            "speed", label="动作速度", min_value=0.25, max_value=3, step=0.25, value=1, unit="倍",
            on_change=lambda value: setattr(self, "motion_speed", value),
        )
        buttons = (
            ("all", "完成整套动作（9步）", "all"),
            ("load", "① 夹面放进捞网（5步）", "load"),
            ("cook", "② 捞网入锅煮面（2步）", "cook"),
            ("serve", "③ 提网倒回原盘（2步）", "serve"),
            ("reset", "重置场景", "reset_scene"),
        )
        for key, label, skill in buttons:
            self.panel.add_button(key, label=label, on_click=lambda skill=skill: self.submit(skill))
        self.panel.add_button("stop", label="停止动作", on_click=self.cancel, enabled=False)

    async def _connect(self):
        try:
            self.loop = asyncio.get_running_loop()
            self.close_event = asyncio.Event()
            async with System.connect(
                self.endpoint, env_id=self.env_id, bindings=self.bindings,
            ) as system:
                self.system = system
                await system.snapshot(node="wrs")
                self.ready.set_result(None)
                await self.close_event.wait()
        except Exception as error:
            if not self.ready.done():
                self.ready.set_exception(error)
            raise

    async def _run(self, skill, options):
        snapshot = await self.system.snapshot(node="wrs")
        if snapshot.admission == "HELD" and snapshot.stop_confirmed:
            receipt = await self.system.allow_actions(node="wrs")
            if not receipt.accepted:
                raise RuntimeError("停止状态尚未确认，请稍后重试")
        if skill == "reset_scene":
            handle = await self.system.action(skill)
        else:
            steps = noodle_steps(snapshot.data.objects, group=skill, **options)
            handle = await self.system.start(*steps)
        self.task_handle = handle
        return await handle.wait(timeout=RECIPE_TIMEOUT)

    def submit(self, skill):
        if self.job is not None and not self.job.done():
            return
        options = {}
        if skill != "reset_scene":
            options["motion_speed"] = self.motion_speed
            options["seconds"] = self.cook_seconds
        self.panel.set_value("result", "正在执行")
        self.job = asyncio.run_coroutine_threadsafe(self._run(skill, options), self.loop)

    async def _cancel(self, action_id):
        handle = getattr(self, "task_handle", None)
        if self.job is not None and not self.job.done() and handle is not None:
            return await handle.cancel()
        snapshot = await self.system.snapshot(node="wrs")
        if snapshot.active_action != action_id:
            return
        return await self.system.clients["wrs"].cancel(ControlRequest(
            interrupt_id=new_id(), boot_id=snapshot.boot_id,
            control_epoch=snapshot.control_epoch, action_id=action_id,
        ))

    def cancel(self):
        if self.active_action:
            future = asyncio.run_coroutine_threadsafe(
                self._cancel(self.active_action), self.loop,
            )
            # Read the result on the next drawing tick so RPC failures are visible.
            self.cancel_job = future
            self.panel.set_value("result", "正在停止")

    def update(self, snapshot):
        facts = snapshot.data.facts
        self.active_action = snapshot.active_action
        roles = role_ids(snapshot.data.objects)
        food = snapshot.data.objects[roles["food"]]
        loaded = food.location == roles["strainer"]
        held = snapshot.data.robot.held_object
        completed = food.location == roles["surface"]
        self.panel.set_value("stage", facts.get("manipulation_stage", "等待状态"))
        places = {"plate": "盘中", "gripper": "夹爪中", "air": "转移中",
                  "skimmer": "捞网中"}
        places.update({roles["surface"]: "盘中", roles["strainer"]: "捞网中", None: "原盘中"})
        self.panel.set_value("food", places.get(food.location, "待确认"))
        width = snapshot.data.robot.kinematics.gripper_width
        self.panel.set_value("gap", "未建模" if width is None else f"{width*1000:.1f} 毫米")
        self.panel.set_value(
            "timer", f"{facts.get('hold_elapsed_seconds', 0):.1f} / "
                     f"{facts.get('hold_seconds', 10):g} 秒",
        )
        if self.job is not None and self.job.done():
            try:
                result = self.job.result()
                message = {
                    ActionState.SUCCEEDED: "动作完成",
                    ActionState.CANCELLED: "已停止，请重置后再开始",
                }.get(result.state, "动作未完成，请检查位置后重置")
                self.panel.set_value("result", message)
            except Exception:
                self.panel.set_value("result", "操作失败，请检查节点连接或重置场景")
            self.job = None
        cancel_job = getattr(self, "cancel_job", None)
        if cancel_job is not None and cancel_job.done():
            try:
                receipt = cancel_job.result()
                if receipt is not None and not receipt.accepted:
                    self.panel.set_value("result", "停止未受理，请重试")
            except Exception:
                self.panel.set_value("result", "停止未确认，请检查节点连接")
            self.cancel_job = None
        idle = snapshot.active_action is None and self.job is None and snapshot.stop_confirmed
        action_idle = (idle and snapshot.admission == "OPEN"
                       and not facts.get("manipulation_interrupted", False))
        for key in ("all", "load"):
            self.panel.set_enabled(key, action_idle and held is None
                                   and not loaded and not completed)
        self.panel.set_enabled("cook", action_idle and loaded and held is None)
        self.panel.set_enabled("serve", action_idle and loaded and held == roles["strainer"]
                               and facts.get("timed_hold_complete", False))
        self.panel.set_enabled("reset", idle)
        self.panel.set_enabled("seconds", idle)
        self.panel.set_enabled("speed", idle)
        self.panel.set_enabled("stop", snapshot.active_action is not None)

    def close(self):
        # Closing an observer does not send a robot stop or cancel a remote action.
        if not self.worker.done():
            self.loop.call_soon_threadsafe(self.close_event.set)
        try:
            self.worker.result(timeout=10)
        finally:
            self.pool.shutdown(wait=True)
