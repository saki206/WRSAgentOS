"""对 D405 看到的目标生成抓取候选；先启动 00–05，改 TARGET 后直接运行。"""

import asyncio
import os

from examples._session import use_local_token
from examples.grasping.settings import connection_settings
from wrs_agent import System
from wrs_agent.nodes.camera.client import CameraClient
from wrs_agent.nodes.grasp.data import read_grasps
from wrs_agent.planner.grasp_parts import grasp_preparation_plan
from wrs_agent.skills.vision.grasping import GraspsOutput

ENDPOINT, ENV_ID, SESSION = connection_settings()
TARGET = "banana"  # YOLO-World 使用简短英文目标，例如 cup、green pepper。
SELECTION = "only"  # 多个目标默认拒绝；明确指定时可用 leftmost / rightmost。


async def suggest_part(goal):
    from pydantic import ValidationError

    from wrs_agent.planner.providers.part_grounding import (
        PartGroundingClient,
        part_model_config,
        part_model_key,
        part_settings,
    )
    from wrs_agent.planner.providers.wire import LLMError

    try:
        settings = part_settings(os.environ.get("WRS_PART_LLM_ENV_FILE"))
        config = part_model_config(settings)
        key = part_model_key(config, settings)
        async with PartGroundingClient(config, key) as client:
            suggestion = await client.suggest(goal)
    except LLMError as exc:
        raise SystemExit(f"部位建议失败：{exc.code}") from None
    except ValidationError:
        raise SystemExit("部位建议返回无效结果。") from None
    return suggestion.object, suggestion.preferred_parts[0].part


async def main():
    use_local_token(SESSION)
    async with System.connect(ENDPOINT, env_id=ENV_ID) as system:
        for node, script in (("camera", "01_start_camera.py"), ("yolo", "02_start_locator.py"),
                             ("segmentation", "03_start_fastsam.py"),
                             ("graspnet", "04_start_graspnet.py"), ("agent", "05_start_agent.py")):
            try:
                await system.registry.wait_for(node, timeout=5)
            except TimeoutError:
                raise SystemExit(f"未发现就绪的 {node}，请先运行 {script} 并保持运行。"
                                 "各脚本的 ENDPOINT / ENV_ID / SESSION 需一致。") from None
        camera = CameraClient(system.registry.transport("camera"))
        status = await camera.status()
        if status["state"] != "ready":
            raise SystemExit(f"相机没有新观测：{status['reason']}。请检查 D405 与 01 的输出。")
        if status["observation"]["source"] != "realsense":
            raise SystemExit("本例需要 RealSense 实际观测，请运行 01_start_camera.py。")

        goal = os.environ.get("WRS_GRASP_GOAL", "").strip()
        if len(goal) > 1024:
            raise SystemExit("WRS_GRASP_GOAL 不得超过 1024 字符。")
        # Check the advertised skill before calling the model or submitting the task.
        skill = "segment_part" if goal else "segment_region"
        skills = system.registry.snapshot().get("segmentation", {}).get("skills", {})
        if skill not in skills:
            required = "1" if goal else "0"
            raise SystemExit(f"03 未提供 {skill}，请用 WRS_PART_GROUNDING={required} 重启 03。")
        target, part = await suggest_part(goal) if goal else (TARGET, None)
        plan = grasp_preparation_plan(target, selection=SELECTION, goal=goal, part=part)
        task = await system.start(*plan.steps)
        result = await task.wait(timeout=90)
        print("感知任务结果：", result.state)
        if result.state != "SUCCEEDED":
            raise SystemExit(f"未生成抓取候选：{result.reason}")

        # 只读取本任务的 infer_grasps 动作，不能拿另一次任务的最新结果。
        history = (await system.status())["action_history"]
        actions = [item for item in history
                   if item["task_id"] == task.id and item["skill"] == "infer_grasps"]
        if len(actions) != 1:
            raise SystemExit("找不到本任务唯一的抓姿动作记录，请重新运行本例。")
        action = await system.clients["graspnet"].status(actions[0]["action_id"])
        if action.state != "SUCCEEDED":
            raise SystemExit("本任务的抓姿动作已失效，请重新运行本例。")
        output = GraspsOutput.model_validate(action.output)
        data, age = await read_grasps(system.registry.transport("graspnet"),
                                      output.candidate_set_id, task.id)
        observation = data.info.segment.observation
        if (observation.source != "realsense" or observation.observation_id != output.observation_id
                or len(data.info.candidates) != output.count):
            raise SystemExit("抓姿结果与本次 RealSense 观测不一致。")
        best = data.info.candidates[0]
        if part is not None:
            provenance = data.info.segment.part
            if provenance is None or provenance.name != part:
                raise SystemExit("抓姿结果与请求的部位不一致。")
            print("部位：", provenance.name, "；分割 ID：", data.info.segment.segment_id)
        print("目标：", target, "；观测：", observation.observation_id)
        print(f"候选数：{output.count}；观测年龄：{age:.2f} 秒")
        print("相机坐标系：", observation.frame_id)
        print("首选位置（米）：", best.translation)
        print(f"开口：{best.width_m * 1000:.1f} mm；模型质量分数：{best.score:.4f}")
        print("以上为未做场景碰撞检查的候选；本例不执行机械臂动作。")


if __name__ == "__main__":
    asyncio.run(main())
