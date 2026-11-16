# Simple TTS Workflow

一个运行在本地的 TTS Web 工作流。项目以 **VoxCPM2 可控声音克隆**为默认入口，同时提供 **Qwen3.5 情绪分析 + VoxCPM2 情景配音**、VoxCPM2 声音设计与 Hi-Fi 克隆，以及 Qwen3-TTS 的声音克隆和 VoiceDesign 流程。

输入多行文本后，程序会为每个非空行生成一个独立的 `.wav` 文件，并把本次任务的参数、参考素材和情绪分析结果写入 `metadata.json`。

## 当前功能

- 7 种生成模式，覆盖声音克隆、情景配音和声音设计。
- 支持批量生成、网页试听、单文件下载和深色模式。
- 支持单条多次重新生成、候选版本试听筛选，以及只保留选中版本。
- 独立设置页可调整重新生成次数，默认 3 次。
- 自动保存当前结果和候选版本，刷新或下次打开后可以继续筛选。
- 支持从 `role/` 加载本地声音预设，也可以临时上传参考音频。
- 情景配音支持自动识别和提示词辅助两种分析方式。
- 模型按需加载；启动 Web 服务本身不会立即下载模型。
- Hugging Face 下载支持镜像、失败重试、官方源回退和离线缓存。
- 输出按任务分目录保存，并记录可追溯的 JSON 元数据。

## 运行环境

- Python `3.12`（项目限制为 `>=3.12,<3.13`）
- 推荐 Windows + NVIDIA GPU；CPU 可以运行 Qwen3-TTS，但 1.7B 模型生成会很慢
- 首次使用某个流程时，需要下载对应模型并占用相应磁盘空间

当前 `uv.lock` 锁定了以下硬件相关构建：

- PyTorch `2.10.0+cu130`
- `llama-cpp-python 0.3.34` Vulkan wheel
- VoxCPM `2.0.2`
- Qwen TTS `0.1.1`

因此，仓库中的默认 `uv sync` 更适合已经具备可用 NVIDIA 驱动和 Vulkan 环境的 Windows 机器。若需要 CPU、其他 CUDA 版本或其他操作系统，请根据目标环境调整 `pyproject.toml` 中的包源并重新生成锁文件。

## 安装与启动

推荐使用 [uv](https://docs.astral.sh/uv/)：

```powershell
uv sync
uv run python main.py
```

也可以直接启动 Uvicorn：

```powershell
uv run uvicorn simplettsworkflow.app:app --host 127.0.0.1 --port 8000
```

启动后打开 <http://127.0.0.1:8000>。

已有合适 Python 3.12 环境时，也可以执行：

```powershell
pip install -e .
python main.py
```

> `pip install -e .` 不使用 `uv.lock`，最终安装的硬件相关 wheel 可能与仓库锁定环境不同。

## 快速使用

1. 在左侧选择生成模式；默认是“可控克隆”。
2. 需要声音来源时，选择 `role/` 中的预设或上传参考音频。
3. 填写目标文本。普通模式下，每个非空行生成一个音频。
4. 按当前模式填写参考文本、情绪与语气或其他必填项。
5. 点击“开始生成”，结果可在网页中试听和下载。

VoxCPM2 模式还可以展开高级参数：

- `CFG`：界面允许 `1.0–3.0`，默认 `2.0`
- `推理步数`：界面允许 `4–30`，默认 `10`
- `文本规范化`：将 `normalize=true` 传给 VoxCPM2
- `参考音频降噪`：将 `denoise=true` 传给 VoxCPM2

## 重新生成与候选版本

生成完成后，进入左侧“生成结果”页面：

1. 点击某条音频卡片中的“重新生成 3 次”，只为这一条连续生成 3 个新候选，不重做其他条目。原版本会保留，因此首次操作后共有 4 个版本。
2. 点击“版本 1”“版本 2”等按钮，切换该条的选中版本；播放器和下载链接会随之切换，可逐个试听比较。
3. 可以多次点击重新生成，继续积累候选。新增版本不会自动替换当前选择；每条音频可选中一个版本。
4. 全部选好后，点击“只保留选中版本”。该按钮在存在多个候选时出现，会保留各条选中的音频，**实际删除当前结果中未选中的候选 WAV 文件**，并同步更新元数据。

页面顶部的“重新生成”会将本次整批内容重新生成一遍，并为每条追加一个候选。设置中的次数只用于音频卡片上的单条重生成操作。

重新生成沿用本次提交的文本、模式、音色来源及生成参数；情景配音还会复用当前选中版本的情绪指令，跳过 Qwen 情绪分析，避免重新分析导致语速、音量和表演方式变化。提示词辅助配音会保留该条原句、描述和关键词。同一轮多次重生成使用相同指令，但合成仍有随机差异。生成期间会显示进度并禁用重复提交。如果某次失败，会停止后续尝试，保留原版本和已经成功生成的候选。

### 设置页

点击侧边栏“设置”，在独立页面修改“每条音频重新生成次数”，然后点击“保存设置”。默认值为 **3**，允许 **1–20** 的整数；次数越多，等待时间越长。设置保存在当前浏览器，重新打开仍会沿用。点击“返回”可回到进入设置前的页面。

### 结果自动保存与恢复

每次生成候选、切换选中版本或完成筛选后，页面会自动保存当前结果列表、全部候选、选中状态，以及重新生成所需的提交参数和上传的参考音频。刷新页面或下次打开，会恢复上次结果，可继续试听、下载、重新生成和筛选。

- 使用同一浏览器配置和同一站点地址访问；`localhost` 与 `127.0.0.1`、不同端口之间不共享记录。
- 结果记录和上传参考音频保存在浏览器 IndexedDB，生成的 WAV 文件仍保存在项目 `output/` 中。恢复试听需要本地服务运行且原音频文件仍存在。
- 当前保存的是最近一组结果，不是历史任务列表。开始新的生成任务并成功后，会替换当前列表；旧任务的磁盘文件仍保留。
- “清空结果”会同时清空页面和已保存的结果记录，但不会删除磁盘音频；“只保留选中版本”则会删除未选中的候选文件。
- 清除浏览器站点数据或使用不保留数据的隐私模式，会丢失浏览器中的记录。保存失败时页面会提示，请先下载需要的音频。
- 服务使用 `.local-state/candidate-key` 校验候选文件清理请求，重启后沿用该密钥。保留此文件，才能继续清理重启前保存的候选；该目录已排除在 Git 提交之外。

## 生成模式

| 页面名称 | API `mode` | 引擎 | 参考音频 | 参考文本 | 情绪/语气 |
| --- | --- | --- | --- | --- | --- |
| 可控克隆 | `vox_controllable_clone` | VoxCPM2 | 必需 | 不需要 | 可选，作为 VoxCPM2 控制前缀 |
| 情景配音 | `scene_dubbing` | Qwen3.5 + VoxCPM2 | 必需 | 不需要 | 由 Qwen3.5 逐条生成 |
| Hi-Fi 克隆 | `vox_hifi_clone` | VoxCPM2 | 必需 | 必需 | 不支持，界面会禁用 |
| 参考音频克隆 | `clone` | Qwen3-TTS Base | 必需 | 必需 | 不传给模型，表达主要来自参考音频 |
| 语音设计 | `vox_design` | VoxCPM2 | 不需要 | 不需要 | 可选 |
| 语气设计 | `voice_design` | Qwen3-TTS VoiceDesign | 不需要 | 不需要 | 必需 |
| 设计后复用 | `voice_design_then_clone` | Qwen3-TTS VoiceDesign + Base | 不需要 | 需要设计参考文本 | 必需 |

默认模型：

- VoxCPM2：`openbmb/VoxCPM2`
- Qwen3-TTS Base：`Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- Qwen3-TTS VoiceDesign：`Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- 情绪分析：`bartowski/Qwen_Qwen3.5-2B-GGUF` 中的 `Qwen_Qwen3.5-2B-Q4_K_M.gguf`

### 情景配音

情景配音先用本地 Qwen3.5 GGUF 模型生成结构化的表达指令，再释放 llama.cpp 模型，最后加载 VoxCPM2 完成配音，避免两个模型长期同时占用显存。

#### 自动识别

`scene_dubbing_mode=auto`。每个非空行作为一条原句，Qwen3.5 自动分析情绪、强度、语速、音高、音量以及节奏与停顿。

```text
我们终于赶上了！
我还是没能等到你。
```

#### 提示词辅助

`scene_dubbing_mode=assisted`。每个条目必须严格包含三行：

1. 原句
2. 情感描述
3. `keyword：` 开头的关键词列表

多个条目之间至少留一个空行：

```text
Alright, I should probably fix the filter first…
体现疲惫与无奈，但保持稳定，不破音。
keyword：疲惫，无奈，认命，倦怠

我们终于赶上了！
体现松了一口气后的惊喜，语速稍快。
keyword：惊喜，释然，兴奋
```

情感描述和关键词只用于生成表达指令，不会改写原句。生成指令会显示在结果旁，并保存到 `metadata.json`。

## 声音预设

克隆类模式可以从 `role/` 读取声音预设。每个预设放在一个独立子目录中：

```text
role/
  alice/
    preset.json
    alice.mp3
```

`preset.json` 格式：

```json
{
  "name": "Alice",
  "reference": "alice.mp3",
  "reference_text": "这里填写参考音频中实际说出的完整内容。"
}
```

规则：

- 配置文件必须是预设子目录中的 `.json` 文件。
- `name`、`reference` 和 `reference_text` 都不能为空。
- `reference` 相对当前预设目录解析，且不能指向 `role/` 之外。
- 支持 `.wav`、`.mp3`、`.flac`、`.m4a` 和 `.ogg`。
- 重名、配置无效或参考音频不存在的预设会被跳过，并在终端记录原因。
- 预设参考文本在可控克隆和情景配音中不会参与模型调用，但仍是当前预设格式的必填字段。

仓库提供了 `role/示例预设/` 作为模板；补充真实参考音频后才会出现在页面下拉框中。

## HTTP API

服务提供以下主要入口：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/` | Web 界面 |
| `GET` | `/api/voice-presets` | 返回当前可用声音预设 |
| `POST` | `/api/generate` | 以 `multipart/form-data` 发起生成 |
| `POST` | `/api/results/retain` | 按候选标识保留选中音频并删除指定的其他候选 |

最小的可控克隆请求：

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/generate" `
  -F "mode=vox_controllable_clone" `
  -F "ref_audio=@C:\path\to\reference.wav" `
  -F "texts=第一行要生成的内容。`n第二行要生成的内容。" `
  -F "emotion_instruction=温柔、克制，语速稍慢"
```

`POST /api/generate` 接受以下表单字段：

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `mode` | `vox_controllable_clone` | 生成模式，取值见上方模式表 |
| `scene_dubbing_mode` | `auto` | 情景配音分析方式：`auto` 或 `assisted` |
| `voice_preset` | 空 | 预设名称；设置后优先于上传文件 |
| `ref_audio` | 空 | 参考音频文件 |
| `ref_text` | 空 | 参考音频逐字文本 |
| `texts` | 空 | 目标文本或辅助配音条目 |
| `language` | `Auto` | Qwen3-TTS 使用的语言 |
| `emotion_instruction` | 空 | 情绪与语气描述 |
| `design_ref_text` | 空 | “设计后复用”的参考文本 |
| `cfg_value` | `2.0` | VoxCPM2 CFG |
| `inference_timesteps` | `10` | VoxCPM2 推理步数 |
| `normalize` | `false` | 是否启用文本规范化 |
| `denoise` | `false` | 是否对参考音频降噪 |
| `scene_analyses` | 空 | 情景配音重生成时复用的指令 JSON 数组，每项含 `text`、`instruction`；顺序、条数和原句须与目标文本一致，不传则重新分析 |

成功响应包含输出目录、音频列表和可选的情绪分析结果：

```json
{
  "output_dir": "C:\\path\\to\\output\\20260731-120000",
  "items": [
    {
      "index": 1,
      "text": "要生成的内容",
      "filename": "line_001.wav",
      "path": "C:\\path\\to\\output\\20260731-120000\\line_001.wav",
      "url": "/outputs/20260731-120000/line_001.wav",
      "candidate_id": "<服务返回的候选标识>"
    }
  ],
  "emotion_analyses": []
}
```

## 配置

生成次数偏好通过网页“设置”页保存；以下模型、下载和日志配置通过环境变量设置。

### Hugging Face 下载

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `HF_ENDPOINT` | `https://hf-mirror.com` | 首选下载端点 |
| `SIMPLETTS_HF_FALLBACK_ENDPOINT` | `https://huggingface.co` | 默认镜像失败时的备用端点；显式指定其他 `HF_ENDPOINT` 后默认不回退 |
| `HF_HUB_ETAG_TIMEOUT` | `60` | 元数据请求超时秒数 |
| `HF_HUB_DOWNLOAD_TIMEOUT` | `300` | 文件下载网络超时秒数 |
| `SIMPLETTS_HF_MAX_WORKERS` | `2` | 快照并发下载数 |
| `SIMPLETTS_HF_RETRIES` | `3` | 每个端点的完整下载尝试次数 |
| `HF_HUB_OFFLINE` | 未启用 | 设为 `1` 时只使用本地缓存 |

切换到 Hugging Face 官方源：

```powershell
$env:HF_ENDPOINT = "https://huggingface.co"
uv run python main.py
```

模型下载完成后强制离线运行：

```powershell
$env:HF_HUB_OFFLINE = "1"
uv run python main.py
```

### 模型与运行参数

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `VOXCPM_MODEL` | `openbmb/VoxCPM2` | VoxCPM2 模型 ID 或本地目录 |
| `VOXCPM_OPTIMIZE` | `true` | 是否启用 VoxCPM2 `torch.compile` 优化 |
| `VOXCPM_LOAD_DENOISER` | `false` | 是否加载 VoxCPM2 denoiser |
| `QWEN_TTS_MODEL` | Qwen3-TTS Base 1.7B | Base 模型 ID 或本地目录 |
| `QWEN_TTS_VOICE_DESIGN_MODEL` | Qwen3-TTS VoiceDesign 1.7B | VoiceDesign 模型 ID 或本地目录 |
| `QWEN_TTS_DEVICE` | 自动检测 | 设为 `cuda` 可强制 Qwen3-TTS 使用 GPU |
| `QWEN_TTS_FLASH_ATTENTION` | `1` | 设为 `0` 禁用 FlashAttention；仅在已安装 `flash_attn` 时启用 |

当前锁定的 `voxcpm 2.0.2` 不接受 `device` 参数。即使设置 `VOXCPM_DEVICE`，程序也只会记录警告并忽略它。

### 情绪分析

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `QWEN_EMOTION_MODEL_PATH` | 空 | 本地 GGUF 文件；设置后不从 Hub 下载 |
| `QWEN_EMOTION_MODEL_REPO` | `bartowski/Qwen_Qwen3.5-2B-GGUF` | GGUF 仓库 |
| `QWEN_EMOTION_MODEL_FILE` | `Qwen_Qwen3.5-2B-Q4_K_M.gguf` | 仓库内文件名 |
| `QWEN_EMOTION_N_CTX` | `4096` | llama.cpp 上下文长度 |
| `QWEN_EMOTION_N_GPU_LAYERS` | `-1` | 卸载到 GPU 的层数，`-1` 表示全部 |
| `QWEN_EMOTION_MAIN_GPU` | `1` | llama.cpp Vulkan 设备编号；请按本机枚举顺序调整 |

### 日志

`SIMPLETTS_LOG_LEVEL` 默认为 `INFO`。终端会记录请求、模型下载与加载、情绪分析、逐行生成、文件写入和耗时。查看静态资源请求等更详细信息：

```powershell
$env:SIMPLETTS_LOG_LEVEL = "DEBUG"
uv run python main.py
```

## 输出文件

每次生成都会创建新的时间戳目录；同一秒内的任务会自动追加序号：

```text
output/
  20260731-120000/
    line_001.wav
    line_002.wav
    metadata.json
  20260731-120000-01/
    line_001.wav
    metadata.json
```

临时上传的参考音频保存在 `.tmp_uploads/`。每次单条重新生成也会创建新的输出目录，候选版本可能分布在多个目录中；结果页显示选中版本所在目录，支持复制路径。筛选操作会删除未选中的候选音频，但不会将保留的文件合并到同一目录。程序不会自动清理其他历史输出或上传文件，需要时请自行归档或删除。

## 项目结构

```text
.
├── main.py                         # 本地启动入口
├── pyproject.toml                  # 项目与依赖配置
├── simplettsworkflow/
│   ├── app.py                      # FastAPI 路由与请求处理
│   ├── candidates.py               # 候选标识校验与文件清理
│   ├── tts.py                      # VoxCPM2 / Qwen3-TTS 工作流
│   ├── emotion.py                  # Qwen3.5 GGUF 情绪分析
│   ├── model_download.py           # Hugging Face 下载与缓存
│   ├── voice_presets.py            # 声音预设加载
│   ├── logging_config.py           # 日志配置
│   └── settings.py                 # 路径和默认模型
├── templates/index.html            # Web 页面
├── static/                         # 前端脚本与样式，含 IndexedDB 结果存储
├── role/                           # 本地声音预设
├── output/                         # 生成结果
└── tests/                          # Python 与 Node.js 测试
```

## 测试

测试使用假模型验证 API、模式分派、预设解析、情绪分析、下载重试和元数据写入，不会实际执行大模型推理：

```powershell
uv run pytest
```

前端流程测试需要 Node.js，覆盖多次生成、候选筛选、次数设置和结果恢复，不调用模型：

```powershell
node --test tests/test_candidates_frontend.cjs tests/test_results_persistence.cjs
```

## 注意事项

- Qwen3-TTS Base 的克隆接口没有独立的 `instruct` 参数。不要把“用伤感的语气说”等提示拼到目标文本前，否则它会被当作正文朗读。
- VoxCPM2 Hi-Fi 克隆路径会忽略语气控制，因此该模式在界面中禁用情绪与语气字段。
- 模型会在首次使用对应模式时下载；Qwen3-TTS 会先下载完整仓库快照，再从本地路径加载。
- 本服务默认只监听 `127.0.0.1`，没有用户认证。若修改为公网监听，请自行增加访问控制、上传限制和反向代理。

## 参考

- [VoxCPM2 Quick Start](https://voxcpm.readthedocs.io/en/latest/quickstart.html)
- [VoxCPM2 Usage Guide](https://voxcpm.readthedocs.io/en/latest/usage_guide.html)
- [VoxCPM2 API Reference](https://voxcpm.readthedocs.io/en/latest/reference/api.html)
- [Qwen3-TTS 1.7B Base 模型卡](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- [Qwen3-TTS 1.7B VoiceDesign 模型卡](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
