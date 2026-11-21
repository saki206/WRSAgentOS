# 通用抓取部位分割：GPT + FastSAM

给程序一张照片和一条抓取口令。第一次 GPT 调用根据文字提出目标和首选部位，第二次结合原口令和整张图片核实部位归属，输出 0–999 归一化框。物体和部位来自模型返回值。视觉提示词要求区分会随物体移动的固定附件与独立支撑架，无法从图像确认时返回空框。

FastSAM 生成候选掩码，按候选外接框与 GPT 框的 IoU 选择。整图无匹配时依次尝试外扩裁剪、降低置信度后的整图和裁剪图。初次整图匹配有歧义时直接返回失败。裁剪结果还原为原图尺寸。

## 运行

使用仓库要求的 Python 3.12。以下命令从仓库根目录运行，先激活自己的 Python 环境。默认使用 CUDA；已有可用 CPU 版 PyTorch 时可以显式传入 `--device cpu`。

```powershell
python -m pip install -e ".[camera,yolo]"
python scripts/download_fastsam.py --download
codex login status
python -m scripts.check_gpt_fastsam --image "photo.jpg" --goal "把碗夹起来，抓把手" --output-dir ".local/runs/bowl-01"
```

联机模式通过已认证的 Codex CLI 发送口令和第二阶段的图片。默认模型为历史实验使用的 `gpt-6.1-sol`，可用 `--model` 指定账号可用的图像模型。本入口沿用 Codex CLI 认证。每次指定一个新的输出目录。

使用已有的、与输入图对应的定位结果离线复放时，加上 `--box-json <gpt-box.json>`，口令用于记录，两个 GPT 调用都会跳过。

| 输出 | 内容 |
| --- | --- |
| `input.png` | 统一方向后的 RGB 输入 |
| `grasp-part-suggestion.json` | GPT 建议的目标、首选部位、理由与条件，联机模式生成 |
| `grounding-prompt.txt` | 本次填入目标和首选部位的视觉提示词，联机模式生成 |
| `gpt-box.json` | 归一化定位框、状态与可见证据 |
| `fastsam.json` | 像素坐标、匹配状态、IoU 或失败原因 |
| `fastsam-final-mask.png` | 成功时的原图尺寸黑白掩码 |
| `fastsam-final-overlay.jpg` | 青色定位框、红色掩码的检查图 |

退出码 0 表示成功选择掩码。退出码 1 表示部位未核实、掩码匹配失败或程序运行失败。该入口输出二维结果。

## 对照老师仓库逐文件核查

基线是 `chenhaox/WRSAgentOS` 的 `main` 提交 `80952866c1da69819f520bd34f891ff0aa28eac6`。

| 文件 | 类型 | 检查点 |
| --- | --- | --- |
| `wrs_agent/planner/grasp_part_prompts.py` | 新增 | 文字部位建议、视觉归属核实提示词，以及返回字段、空框状态和坐标范围校验。 |
| `wrs_agent/nodes/segmentation/fastsam.py` | 修改 | 保留现有框匹配 API，增加可显式调用的局部重试。复用上游现有的 IoU 选择器。 |
| `scripts/check_gpt_fastsam.py` | 新增 | 接收口令和图像，调用 Codex，根据框调用 FastSAM，保存定位、掩码和叠图。可用已有框离线复放。 |
| `examples/grasping/README.md` | 修改 | 为抓取示例增加离线分割的文档入口。 |
| `docs/GPT_FASTSAM.md` | 新增 | 运行方法、文件交接、逐文件审查清单。 |

历史离线实验采用当时的具体提示词以及已知测试图。通用入口将物体和部位改为动态生成，历史结果不能作为这条新入口的准确率。
