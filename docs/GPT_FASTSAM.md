# 通用抓取部位分割：GPT + FastSAM

给程序一张照片和一条抓取口令。第一次 GPT 调用根据文字提出目标和首选部位，第二次结合原口令和整张图片核实部位归属，输出 0–999 归一化框。物体和部位来自模型返回值。视觉提示词要求区分会随物体移动的固定附件与独立支撑架，无法从图像确认时返回空框。

例如，口令“夹起方盒，夹侧板”会将方盒和侧板作为视觉核实目标。

FastSAM 按候选外接框与 GPT 框的 IoU 匹配，最低 IoU 为 0.50，前两名至少相差
0.05。默认置信度为 0.4。整图无匹配时依次尝试外扩裁剪、置信度 0.25 的整图和裁剪图。
任意一轮遇到歧义就停止。裁剪结果还原为原图尺寸。

默认要求框的宽、高各至少 2 像素。可用 `--min-box-pixels` 和
`--max-box-fraction` 控制尺寸与占图面积比例。面积上限默认为 1，允许目标部位占据整图。

## 运行

使用仓库要求的 Python 3.12。以下命令从仓库根目录运行，先激活自己的 Python 环境。默认使用 CUDA；已有可用 CPU 版 PyTorch 时可以显式传入 `--device cpu`。

```powershell
python -m pip install -e ".[camera,yolo,llm]"
python scripts/download_fastsam.py --download
```

在本地创建 `.env.openai`，把占位符换成你的 OpenAI API Key。Git 已忽略 `.env.*` 文件。

```dotenv
OPENAI_API_KEY=your_openai_api_key
```

```powershell
python -m scripts.check_gpt_fastsam --image "photo.jpg" --goal "把碗夹起来，抓把手" --env-file ".env.openai" --output-dir ".local/runs/bowl-01" --save-images
```

默认使用 OpenAI Responses API、`gpt-6.2-sol` 和 `medium` 推理档。两个请求共享
120 秒时限，可用 `--timeout` 调整。可以通过 `--model`、`--base-url`、
`--protocol`、`--reasoning-effort`、`--max-output-tokens` 和 `--proxy` 配置。
支持 Responses 和 Chat Completions 协议，服务须支持图像输入及严格 JSON Schema。
自定义地址使用 `LLM_API_KEY`，OpenAI 默认地址使用 `OPENAI_API_KEY`。
环境变量优先于 `--env-file`，系统代理变量不参与请求。

## 图像与留存

联机定位会将经过方向校正、重新编码的整张 RGB 图发送给配置的服务。现场照片应在提交前
遮盖无关人员、铭牌和屏幕。请求设置 `store: false`，服务端留存仍受服务及账号配置影响。
OpenAI 默认滥用监控日志可能保留最多 30 天，详见 [数据控制说明](https://developers.openai.com/api/docs/guides/your-data)。
默认只保存 JSON 和黑白掩码。传入 `--save-images` 才保存输入图与彩色叠图。

使用已有的、与输入图对应的定位结果离线复放时，加上 `--box-json <gpt-box.json>`，口令用于记录，两个 GPT 调用都会跳过。

| 输出 | 内容 |
| --- | --- |
| `input.png` | 统一方向后的 RGB 输入，需 `--save-images` |
| `grasp-part-suggestion.json` | GPT 建议的目标、首选部位、理由与条件，联机模式生成 |
| `gpt-box.json` | 归一化定位框、状态与可见证据 |
| `fastsam.json` | 像素坐标、匹配状态和逐轮搜索范围、置信度、候选数及选中 IoU 或拒绝原因 |
| `fastsam-final-mask.png` | 成功时的原图尺寸黑白掩码 |
| `fastsam-final-overlay.jpg` | 青色定位框、红色掩码的检查图，需 `--save-images` |

结果完整写入临时目录后才发布至输出路径。空框、拒框或无匹配会保存完整诊断。
退出码 0 表示选中掩码，1 表示未选中掩码或其他运行失败，2 表示命令参数或配置错误，
3 表示模型请求失败或结构化结果无效。

## 用独立标注评估

传入 `--expected-mask <人工标注PNG>` 可在报告中得到 `expected_box_iou` 和
`expected_mask_iou`，分别评估 GPT 框与标注外接框、预测掩码与标注掩码。标注须与方向
校正后的图像尺寸一致，目标部位用白色标记。未选出掩码记为 0。应在未用于调整提示词的
样本上评估。外接框 IoU 仅衡量候选与模型定位框的一致性，像素掩码 IoU 才与人工标注
进行比较。没有独立标注时，GPT 的证据仅供人工复查，`matched` 表示候选匹配成功。
