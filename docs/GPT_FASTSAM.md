# 通用抓取部位分割：GPT + FastSAM

给程序一张照片和一条抓取口令。第一次 GPT 调用根据文字提出目标和首选部位，第二次结合原口令和整张图片核实部位归属，输出 0–999 归一化框。物体和部位来自模型返回值。视觉提示词要求区分会随物体移动的固定附件与独立支撑架，无法从图像确认时返回空框。

例如，口令“夹起方盒，夹侧板”会将方盒和侧板作为视觉核实目标。

FastSAM 生成候选掩码，按候选外接框与 GPT 框的 IoU 选择。整图无匹配时依次尝试外扩裁剪、降低置信度后的整图和裁剪图。初次整图匹配有歧义时直接返回失败。裁剪结果还原为原图尺寸。

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
python -m scripts.check_gpt_fastsam --image "photo.jpg" --goal "把碗夹起来，抓把手" --env-file ".env.openai" --output-dir ".local/runs/bowl-01"
```

若已在环境变量中设置 `OPENAI_API_KEY`，可以省略 `--env-file`。脚本通过 [OpenAI Responses API](https://developers.openai.com/api/reference/resources/responses/methods/create) 发送口令和第二阶段的图片。默认模型为 `gpt-4.1-mini`，可用 `--model` 指定具有图像输入和结构化输出能力的 OpenAI API 模型。每次指定一个新的输出目录。

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
