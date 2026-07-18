# Simple Qwen3-TTS Workflow

一个简单的本地 TTS Web 工作流：默认使用 VoxCPM2 进行可控声音克隆，也保留 Qwen3-TTS 的 clone 与 VoiceDesign 流程。生成的 `.wav` 文件会保存到 `output/`。

## 环境

Qwen3-TTS 会先把完整模型下载到 Hugging Face 本地缓存，再从本地快照加载。这样 Transformers 在检查可选的 custom generation 代码时只访问本地文件，不会在模型加载阶段反复发送 HEAD 请求。

- Python 3.12
- 建议使用 NVIDIA GPU。CPU 可以尝试运行，但 1.7B 模型会很慢。
- 首次运行会下载所选模型权重。VoxCPM2 默认模型是 `openbmb/VoxCPM2`，Qwen3-TTS 默认模型是 `Qwen/Qwen3-TTS-12Hz-1.7B-Base`。

## 安装

```powershell
uv sync
```

如果你已经有合适的 Python 3.12 环境，也可以使用：

```powershell
pip install -e .
```

## 启动

```powershell
uv run uvicorn simplettsworkflow.app:app --host 127.0.0.1 --port 8000
```

然后打开 <http://127.0.0.1:8000>。

也可以运行：

```powershell
uv run python main.py
```

## 使用流程

1. 选择生成模式，默认是 `VoxCPM2 / 可控克隆`。
2. 输入一行或多行目标文本，每一行会生成一个独立音频文件。
3. 根据模式填写参考音频、参考文本或情绪/语气描述。
4. 生成结果默认保存在 `output/YYYYMMDD-HHMMSS/`。

## 语音 Clone 预设

克隆模式可以从 `role/` 文件夹读取语音预设。每个预设单独放在一个子文件夹里，配置文件和参考音频放在同一目录内：

```text
role/
  alice/
    preset.json
    alice.mp3
```

配置文件格式：

```json
{
  "name": "Alice",
  "reference": "alice.mp3",
  "reference_text": "这段话需要和参考音频中实际说出的内容一致。"
}
```

`reference` 支持 `.wav`、`.mp3`、`.flac`、`.m4a` 和 `.ogg`，路径相对当前预设文件夹解析。启动页面后，“可控克隆”“Hi-Fi 克隆（高级）”和“参考音频克隆”模式会显示“语音预设”下拉框。选择预设后，程序会优先使用预设音频和参考文本；如果不选预设，仍然使用上传参考音频的原流程。

## 生成模式

### VoxCPM2

- `可控克隆`：默认模式。上传参考音频克隆音色，可选填写情绪/语气描述。程序会按 VoxCPM2 要求把描述包装成 `(语气描述)目标文本`，并通过 `reference_wav_path` 保留音色。
- `语音设计`：不需要参考音频，可选填写语气描述来生成新声音。
- `Hi-Fi 克隆（高级）`：上传参考音频并填写逐字参考文本，提高声音相似度。VoxCPM2 文档说明这个路径会忽略语气控制，所以界面会禁用情绪/语气描述。

### Qwen3-TTS

- `克隆参考音频`：使用 `Qwen/Qwen3-TTS-12Hz-1.7B-Base`。需要上传参考音频和对应文本。这个模式没有独立 `instruct` 参数，语气主要来自参考音频本身。
- `语气设计`：使用 `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`。不需要参考音频，情绪/语气描述会作为真正的 `instruct` 参数传给模型。
- `设计后复用`：先用 VoiceDesign 根据语气描述生成一段参考音频，再用 Base 模型把它作为 clone prompt 复用，适合多行文本保持同一种设计声音。

注意：不要在克隆模式里把“用伤感的语气说”写到目标文本前面。Base voice clone 会把它当正文朗读；本程序已经避免把语气描述拼进 clone 文本。

## 配置

下载相关配置：

- <code>HF_ENDPOINT</code>：Hugging Face 下载端点，默认 <code>https://hf-mirror.com</code>；如需使用官方源，设置为 <code>https://huggingface.co</code>
- <code>SIMPLETTS_HF_FALLBACK_ENDPOINT</code>：备用下载端点；默认镜像不可用时自动切换到 <code>https://huggingface.co</code>。显式设置其他 <code>HF_ENDPOINT</code> 时默认不自动切源
- <code>HF_HUB_ETAG_TIMEOUT</code>：仓库元数据/HEAD 请求超时秒数，默认 <code>60</code>
- <code>HF_HUB_DOWNLOAD_TIMEOUT</code>：单次文件下载网络超时秒数，默认 <code>300</code>
- <code>SIMPLETTS_HF_MAX_WORKERS</code>：模型快照并发下载数，默认 <code>2</code>；较低并发可减少镜像/CDN 元数据请求偶发失败
- <code>SIMPLETTS_HF_RETRIES</code>：完整快照下载失败后的尝试次数，默认 <code>3</code>；重试会继续复用已经下载的缓存
- <code>HF_HUB_OFFLINE=1</code>：完全离线，只使用已经下载完整的本地缓存

如果当前网络无法访问默认镜像，可以在启动前切换到官方源：

    $env:HF_ENDPOINT = "https://huggingface.co"
    uv run python main.py

模型第一次下载成功后，后续运行会直接复用 Hugging Face 缓存。若要强制禁止所有联网检查：

    $env:HF_HUB_OFFLINE = "1"
    uv run python main.py

- `VOXCPM_MODEL`：VoxCPM2 模型 ID 或本地模型目录，默认 `openbmb/VoxCPM2`
- `VOXCPM_DEVICE`：VoxCPM2 运行设备，默认 `auto`
- `VOXCPM_OPTIMIZE`：是否启用 VoxCPM2 `torch.compile` 优化，默认 `true`
- `VOXCPM_LOAD_DENOISER`：是否加载 VoxCPM2 denoiser，默认 `false`
- `QWEN_TTS_MODEL`：模型 ID 或本地模型目录，默认 `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
- `QWEN_TTS_VOICE_DESIGN_MODEL`：VoiceDesign 模型 ID 或本地模型目录，默认 `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`
- `QWEN_TTS_DEVICE`：设置为 `cuda` 强制使用 GPU；未设置时自动检测 CUDA
- `QWEN_TTS_FLASH_ATTENTION=0`：禁用 FlashAttention 参数

## 输出

每次生成会创建一个新目录：

```text
output/
  20260504-153000/
    line_001.wav
    line_002.wav
    metadata.json
```

`metadata.json` 会记录生成引擎、模式、参考素材、语气描述、Vox 参数和输出文件信息。

## 参考

- [VoxCPM2 Quick Start](https://voxcpm.readthedocs.io/en/latest/quickstart.html)
- [VoxCPM2 Usage Guide](https://voxcpm.readthedocs.io/en/latest/usage_guide.html)
- [VoxCPM2 API Reference](https://voxcpm.readthedocs.io/en/latest/reference/api.html)
- [Qwen3-TTS 1.7B Base 模型卡](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- [Qwen3-TTS 1.7B VoiceDesign 模型卡](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
