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
自定义地址使用 `WRS_PART_LLM_API_KEY`，OpenAI 默认地址使用 `OPENAI_API_KEY`。
旧配置的 `LLM_API_KEY` 只在其配置的 API 地址与部位模型地址一致时可复用。
命令行只提供本地程序的参数。文字建议和图像定位均由 Python 通过 HTTP API Key 调用。
配置优先级依次是命令行参数、进程环境变量、`--env-file`、默认值。系统代理变量不参与请求。

本地图片入口和实时模式共享以下模型配置，可写在同一个本地配置文件里：

| 变量 | 用途 |
| --- | --- |
| `WRS_PART_LLM_MODEL` | 模型 ID，默认 `gpt-6.2-sol` |
| `WRS_PART_LLM_PROTOCOL` | `openai_responses` 或 `openai_chat` |
| `WRS_PART_LLM_BASE_URL` | 服务 API 根地址，默认 `https://api.openai.com/v1` |
| `WRS_PART_LLM_API_KEY` | 自定义部位模型服务的专用密钥 |
| `WRS_PART_LLM_REASONING_EFFORT` | GPT‑6.2 Sol 默认 `medium`。其他模型默认使用服务端设置，`omit` 显式省略 |
| `WRS_PART_LLM_TIMEOUT_S` | 请求时限，实时定位还受同帧观测有效期约束 |
| `WRS_PART_LLM_MAX_TOKENS` | 单次输出 token 上限 |
| `WRS_PART_LLM_PROXY` | 显式 HTTP(S) 代理 |

部位模型只读取这些专用变量。原有任务规划模型继续使用自己的 `LLM_*` 配置。

## 图像与留存

联机定位会将经过方向校正、重新编码的整张 RGB 图发送给配置的服务。现场照片应在提交前
遮盖无关人员、铭牌和屏幕。请求设置 `store: false`，服务端留存仍受服务及账号配置影响。
OpenAI 默认滥用监控日志可能保留最多 30 天，详见 [数据控制说明](https://developers.openai.com/api/docs/guides/your-data)。
默认只保存 JSON 和黑白掩码。传入 `--save-images` 才保存输入图与彩色叠图。
原图文件及重新编码后的 PNG 上限分别为 16 MiB，解码像素数最多 16,777,216。
API 响应和本地 JSON 上限为 64 KiB，超过上限的响应会在解析前中止。

离线复放使用 `--box-json <gpt-box.json>`。程序核对输入图方向校正后的尺寸和 RGB 像素
SHA‑256，两个 GPT 调用都会跳过。旧框没有指纹时，必须显式传入 `--allow-legacy-box`。
结果会标记为未验证的复放，且不能用于标注评估。

| 输出 | 内容 |
| --- | --- |
| `input.png` | 统一方向后的 RGB 输入，需 `--save-images` |
| `grasp-part-suggestion.json` | GPT 建议的目标、首选部位、理由与条件，联机模式生成 |
| `gpt-box.json` | 归一化定位框、状态、可见证据和图像指纹 |
| `input-image.json` | 方向校正后的图像尺寸与像素指纹 |
| `fastsam.json` | 像素坐标、匹配状态和逐轮搜索范围、置信度、候选数及选中 IoU 或拒绝原因 |
| `fastsam-final-mask.png` | 成功时的原图尺寸黑白掩码 |
| `fastsam-final-overlay.jpg` | 青色定位框、红色掩码的检查图，需 `--save-images` |

结果完整写入临时目录后才发布至输出路径。空框、拒框或无匹配会保存完整诊断。
退出码 0 表示选中掩码，1 表示未选中掩码或其他运行失败，2 表示命令参数或配置错误，
3 表示模型请求失败或结构化结果无效。

## 用独立标注评估

传入 `--expected-mask <人工标注PNG>` 可在报告中得到 `expected_box_iou` 和
`expected_mask_iou`，分别评估 GPT 框与标注外接框、预测掩码与标注掩码。
标注时使用方向校正后的图像，目标部位用白色标记。为 `mask.png` 生成 `mask.png.json`，
将当次的输入图指纹和标注文件哈希一起保存：

```powershell
$identity = Get-Content ".local/runs/bowl-01/input-image.json" -Raw | ConvertFrom-Json
@{ image = $identity; mask_sha256 = (Get-FileHash "mask.png").Hash.ToLowerInvariant() } |
    ConvertTo-Json -Depth 4 | Set-Content "mask.png.json" -Encoding utf8
```

使用未参与提示词调试的样本。未选出掩码时，掩码 IoU 为 0。没有独立标注时，
GPT 的证据供人工复查，`matched` 表示候选与框匹配。

## 实时 RGB-D 链路

按[抓取示例](../examples/grasping/README.md)启动 00–05。启动 03 前，在它的进程环境中
设置 `WRS_PART_GROUNDING=1`，并设置 `OPENAI_API_KEY`，或将
`WRS_PART_LLM_ENV_FILE` 指向本地配置文件。默认使用 GPT‑6.2 Sol 中档推理，
会将定位节点保留的一帧 RGB 发送给配置的服务。深度、点云和掩码留在本机。

为运行 06 的进程配置相同的模型和凭据，并设置 `WRS_GRASP_GOAL`，例如
`把碗夹起来，抓把手`。06 从口令取得物体名和首选部位，再提交
`locate_object → segment_part → infer_grasps`。分割节点复用定位保留的 RGB、深度和 XYZ，
在同一像素网格上产生掩码，并通过标准 `segment_id` 交给 GraspNet。
启用部位模式的 03 同时提供 `segment_region`，整物体任务仍可直接使用 YOLO 的框。

节点沿用 30 秒观测有效期，并给分割及下游读取预留 5 秒。定位超时、结果歧义或帧过期时，
本次任务会失败。06 会输出观测 ID、分割 ID 和三维抓姿候选。

原语音示例也可以提交同一份抓姿计划。按[语音部位模式](../examples/voice/README.md#用语音生成指定部位的抓姿)
配置 07 和 03 后，口令来自本地 ASR，抓姿候选通过 09 查看。
