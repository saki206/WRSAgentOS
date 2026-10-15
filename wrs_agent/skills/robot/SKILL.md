---
name: robot
description: Plan observation, named-pose motion, and supported object handling using the available robot skills.
---

# 机器人技能

根据当前可用技能选择动作。`observe` 用于读取当前证据，
`move_named_pose` 用于移动到已知姿态；支持物体操作的节点还可提供
`pick`、`place`、`verify`。具体参数和执行条件见一起提供的 SkillSpec。

把 A 放到 B 的计划通常是：观察，抓取 A，放到 B，检查结果。
用步骤依赖表达顺序；需要决定下一步时使用最新观测。
未找到物体或目标不明确时请求澄清，不能填写想象的坐标。
只有当前能力列表提供相应技能时才可提出该步骤。

WRS 虚拟节点提供观察、命名姿态和 `move_relative`。相对移动以世界坐标系米为单位：上/下为 ±Z，左/右为 ±Y，前/后为 ±X；不是屏幕方向。每次非零位移最多 10 cm，保持末端目标朝向，以关节插值到达目标，不保证笛卡尔直线或避碰。不可达或需切换远处 IK 分支时拒绝执行。`set_gripper` 把 UR7E 末端的 DH50 夹爪完全打开（`open`）或完全闭合（`close`），机械臂不动；闭合只是手指运动，不表示夹住了物体，也不能代替 `pick`。Mock 抓取结果不代表实机接触。
加载了场景的 WRS 虚拟节点还提供 `pick` 和 `place_on`，路径经过碰撞检测。`pick` 只抓场景里正放的长方体：从正上方下降，夹住后抬高 8 cm，观测中的 `held_object` 变为该物体；夹爪张开后两指之间约 2.8 cm，更宽的物体会失败。拿着物体时不能用 `set_gripper`，要用 `place_on` 放下。`place_on` 把手里的物体放到表面正中，表面已有物体或面积不够时失败。抓取成功是虚拟夹持，不代表实机接触。
动作是否成功由节点验证结果确定，不能从“已接收”推断成功。

## 做饭

物体名只用最新观测里出现的名字。容器类物体先确认朝向：倒放的漏网、扣反的锅盖先 `pick`，再 `reorient` 到 `upright`；已经正放的可以省略。
放在表面上（桌面、盘子、灶面）用 `place_on`；放进容器内部或盖上锅用 `seat_in`，不要用 `place_on` 代替，否则会停在边沿上。
煮面示例：`pick(strainer)` → `reorient(strainer, upright)` → `seat_in(strainer, pot)` → `pick(noodles)` → `seat_in(noodles, strainer)`，后一步依赖前一步。
`set_control` 开关灶具，执行时夹爪必须为空。
`pour`、`stir`、`scoop`、`shake_off` 是示教动作，只在节点提供时使用，不能用 `move_relative` 拼出来。

## FAFU 原厨房适配

加载原厨房 mesh 场景后，节点使用原 FAFU 模型，并提供经过该组资产标定的
`pick/reorient/place_on/seat_in/hold_in/pour`。对象名从最新观测取；不能假设为
`strainer/noodles`，原场景使用 `skimmer/plate_block`，也支持按观测中的名称调用。
未标定的物体或物体组合会明确失败，不从外形名称推测抓取点。

`pick` 对面块夹住中心，对漏勺夹住原把手；必须先闭爪、验证几何接触，再提起。
`reorient(..., orientation="upright")` 扶正漏勺并保持抓持。
`place_on(..., placement="working")` 把漏勺放在已标定工作位，释放并撤离；
`placement="initial"` 将它恢复原位置和原横放姿态。默认 `center` 为表面中心。
只有物体/表面组合已标定时才可选择这些放置方式。

`seat_in(object, container)` 放入后释放；`hold_in(object, container, seconds=10)`
持物入容器并保持指定墙钟时间，结束仍握住。持稳等待期间占用 arm；速度参数
不缩短等待。计时完成不代表温度或熟度验证。
`pour(source, target)` 使用原厨房验证过的参数化倾倒轨迹，倒出内容物后扶正并
保持抓持；随后由 `place_on` 归位。这里的轨迹是程序标定，尚无实机示教采集。

动作 `motion_speed` 可选 0.25–3，默认 1；原 DH50 方块抓放也接受该参数。
普通 UR7E 方块场景仍只支持 `place_on` 的默认 `center`，不支持厨房放置位。
`reset_scene` 仅用于显式恢复仿真初始状态。关闭页面不会取消任务。
中途停止保留关节、夹爪间距、物体位姿和持有状态；复位后再运行整套演示。

煮面是一份由 Runtime 执行的九步计划，定义在 `wrs_agent/recipes/noodles.py`；
节点不注册 `cook_noodles` 等整套业务动作，规划器可以单独组合上述基础技能。
本适配验证 IK/关节限位、抓取几何及结果位姿；完整环境碰撞规划、接触力和热学
尚未实现。UR7E 的原方块 `pick/place_on` 保留原 WRS 碰撞规划。
