# Simple Qwen3-TTS Workflow

一个简单的本地 TTS Web 工作流：上传参考音频和参考文本进行 voice clone，或使用 VoiceDesign 通过自然语言控制语气，并把生成的 `.wav` 文件保存到 `output/`。

## 环境

- Python 3.12
- 建议使用 NVIDIA GPU。CPU 可以尝试运行，但 1.7B 模型会很慢。
- 首次运行会下载 Qwen3-TTS 模型权重。也可以提前下载到本地，并通过环境变量指向本地目录。

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

1. 选择生成模式。
2. 输入一行或多行目标文本，每一行会生成一个独立音频文件。
3. 根据模式填写参考音频、参考文本或情绪/语气描述。
4. 生成结果默认保存在 `output/YYYYMMDD-HHMMSS/`。

## 生成模式

- `克隆参考音频`：使用 `Qwen/Qwen3-TTS-12Hz-1.7B-Base`。需要上传参考音频和对应文本。这个模式没有独立 `instruct` 参数，语气主要来自参考音频本身。
- `语气设计`：使用 `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`。不需要参考音频，情绪/语气描述会作为真正的 `instruct` 参数传给模型。
- `设计后复用`：先用 VoiceDesign 根据语气描述生成一段参考音频，再用 Base 模型把它作为 clone prompt 复用，适合多行文本保持同一种设计声音。

注意：不要在克隆模式里把“用伤感的语气说”写到目标文本前面。Base voice clone 会把它当正文朗读；本程序已经避免把语气描述拼进 clone 文本。

## 配置

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

`metadata.json` 会记录生成模式、参考素材、语言、语气描述和输出文件信息。

## 参考

- [Qwen3-TTS 1.7B Base 模型卡](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- [Qwen3-TTS 1.7B VoiceDesign 模型卡](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign)
- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
