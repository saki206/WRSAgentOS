# 原厨房上的通用动作与煮面组合

基线为搭档 `design` 的 `7119da4109ed3ade5c46d4593086bb7ac5b7f91e`，
合入同学 `feature/cooking-scene` 的 `d1162520742f5678f603723c0af2b401dc0aaf47`。
2026-10-15 核对远端时，两者均是对应分支的最新提交。
WRS 子模块保持双方固定的 `7815e6f110fd161fe3c3b7e6f978e5393c3cf502`。

原 FAFU URDF、STL、厨房八物体 `scene.toml`、尺寸和初始位置均保持原版。
没有新增碗或替换夹爪，最终仍倒回原来放面的盘子。

## 启动

使用 Python 3.12，先初始化子模块、核心依赖和 Router：

```powershell
git submodule update --init --recursive
./scripts/bootstrap.ps1
./scripts/install_router.ps1
```

核心 bootstrap 不安装完整 WRS 科学包。随后将本次使用的依赖装入项目目录：

```powershell
$kitchenPython = if ($env:WRS_AGENT_PYTHON) { $env:WRS_AGENT_PYTHON } else { 'python' }
& $kitchenPython -m pip install --target .local/deps `
    'numpy==1.26.4' 'scipy==1.16.2' 'mujoco==3.5.0' 'wgpu==0.32.0' `
    'websockets==15.0.1' 'urdf_parser_py==0.0.4' 'xacro==2.1.1' `
    'rendercanvas==2.7.2' 'glfw==2.10.2' 'pillow==12.3.0' 'opencv-python==4.11.0.86'
./scripts/run.ps1 scripts/doctor.py --probe-wrs
./scripts/run.ps1 examples/wrs/17_start_cooking_node.py
```

看板 http://127.0.0.1:8014/ ，原厨房画面 http://127.0.0.1:8004/ 。
通信端口 7451，环境 `wrs-composed`。与旧版 7449/8000 和搭档比较版 7450/8002
分开，关闭一个版本不会停止另一个版本。127.0.0.1 仅供本机访问。
当前在 Windows 本机验证，未验收干净机器重建或 Linux/macOS。

画面有整套、三个分步、停止、复位、等待时间和运动速度控制。
默认等待 10 秒，运动速度 0.25–3 倍不会缩短等待。
启动时不启用在线模型，不打开相机、麦克风或硬件接口。
关闭观察页面不会取消远端正在运行的动作。

第三个终端可以调用完整流程或分组：

```powershell
./scripts/run.ps1 examples/wrs/16_cook_noodles.py --reset
./scripts/run.ps1 examples/wrs/16_cook_noodles.py --group load --motion-speed 3
./scripts/run.ps1 examples/wrs/16_cook_noodles.py --group cook
./scripts/run.ps1 examples/wrs/16_cook_noodles.py --group serve
```

完整演示后需复位再执行 load。单个基础动作也能从搭档看板的“技能”页调用。

## 合同与实现

沿用 `wrs_agent/skills/robot/definitions.py` 的单份合同、发现、资源占用、
Runtime 步骤依赖、停止确认和 Dashboard。原 UR7E + DH50 的方块抓放与碰撞规划
保留。带原厨房网格的场景选择 FAFU，普通方块场景和未指定场景保留 UR7E。

| 技能 | 原厨房实现与结束状态 |
| --- | --- |
| `pick(object)` | 根据物体标定点接近，实际闭爪并检查接触，提起后保持抓持 |
| `reorient(object, orientation)` | 扶正已夹住的漏勺，结束仍握住 |
| `place_on(object, surface, placement)` | 放在标定表面并松开；支持漏勺工作位、原始横放位和表面中心 |
| `seat_in(object, container)` | 放进已标定容器并释放，更新容器归属；支持面块入网、网入锅 |
| `hold_in(object, container, seconds)` | 新增：持网进入锅内并计时保持，结束仍握住，等待期间占用 arm |
| `pour(source, target)` | 使用标定的倾倒轨迹将内容物倒到原盘，扶正后仍握住来源容器 |
| `set_gripper(command)` | 空手时原夹指开/合，不把闭爪当成抓取成功 |
| `reset_scene()` | 明确恢复初始物体、机器人和原 86 mm 开口，仅用于仿真 |

`motion_speed` 为可选参数，默认 1；`hold_in.seconds` 默认 10，范围 0.1–120。
`place_on.placement` 默认为 `center`，厨房还支持 `working/initial`。普通 UR7E
方块实现明确拒绝这些厨房放置位。不支持的物体、组合或姿态在运动前返回失败。

动作程序不根据名字猜抓取点。`assets/kitchen/actions.toml` 保存原资产的局部抓取
坐标、夹口宽度、开口方向、底部、边缘、容器半径和工作位；按观测的 asset 匹配
配置，因此把对象 ID 从 skimmer 改成另一个名字仍能执行。其他形状或机器人仍需
新的标定/后端，不代表已经支持任意厨具。

原夹指由原 mesh 连通部件在内存中分出，围绕原连接点旋转。原 86 mm 开口先调整
到 40 mm，夹原漏勺收至约 24.5 mm；夹面块收至 60 mm。检查实际夹口和两指接触后
才附着提起，闭爪时补偿接触中心偏移。源模型文件不改动。

## 九步计划

`wrs_agent/recipes/noodles.py` 提供流程模板，传给现有 `system.start(*steps)`：

1. `pick(skimmer)`
2. `reorient(skimmer, upright)`
3. `place_on(skimmer, table, working)`
4. `pick(plate_block)`
5. `seat_in(plate_block, skimmer)`
6. `pick(skimmer)`
7. `hold_in(skimmer, pot, seconds=10)`
8. `pour(skimmer, plate)`
9. `place_on(skimmer, table, initial)`

每一步都是独立 Action，有自己的成功、失败和取消结果。节点没有注册
`cook_noodles` 或三个旧业务阶段 skill；页面的整套/分步按钮只是提交相应步骤计划。
任一步失败后 Runtime 不继续后续步骤，也不在持有 arm 锁时嵌套提交新的动作。

任务取消按搭档 Runtime 的规则先确认停止，再只解除自己建立的停止栅栏；旧任务
不会续跑。关节、夹口、物体位姿和持有关系保持停止时的状态。页面标记中断并要求
显式复位；UNKNOWN 或另一个操作者建立的停止不会被自动解除。

## 验证和边界

```powershell
./scripts/run.ps1 -m pytest tests/unit -q
./scripts/run.ps1 -m pytest tests/integration/test_kitchen_recipe.py tests/integration/test_wrs.py tests/integration/test_fafu_profile.py -q
```

测试覆盖原网格保持、夹口几何接触与桌面间隙、替换对象 ID 后的完整九步、纯规划
无运动副作用、Runtime 九步组合、默认 10 秒等待、错误前置条件、闭爪中取消、持网
等待中取消及全部位姿冻结。原 UR7E 抓放和 FAFU 模型运动回归也单独保留。

本机本次结果：单元测试 840 通过、2 项可选依赖测试跳过；相关集成检查共 16 项
通过，包含 Runtime/模型、原观察器生命周期和 Dashboard。页面整套九步回放成功，
默认等待实际为 10 秒，3 倍速度仅用于加快动作回放。

厨房目前仍使用运动学附着与下落动画，验证 IK/限位、标定接触和目标位姿，未实现
全程碰撞规划、接触力或热学。快照明确 `collision_checked=false`、
`physics_enabled=false`、`thermal_verified=false`。已有参数化倾倒轨迹来自程序标定，
没有宣称采集过实机示教。在线模型和真实感知设备没有在本次验收中调用。

## 文件划分

- `skills/robot/`：沿用并扩展通用合同。
- `env/kitchen_profile.py`、`assets/kitchen/actions.toml`：标定数据及观测 ID 匹配。
- `env/object_motion.py`：共享 IK、轨迹、开合和释放动画。
- `env/kitchen_actions.py`：基础技能执行、持有/容器关系、停止及后置验证。
- `env/fafu_backend.py`、`env/fafu_gripper.py`：原模型后端和原夹指。
- `recipes/noodles.py`：上层九步计划。
- `env/recipe_viewer.py`、`nodes/ui/scene_view.py`：组合按钮与原 Dashboard 的显示适配。
