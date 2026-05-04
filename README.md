# Simple Qwen3-TTS Workflow

一个简单的本地 TTS Web 工作流：上传参考音频和参考文本，输入一行或多行目标文本，使用 `Qwen/Qwen3-TTS-12Hz-1.7B-Base` 进行 voice clone，并把生成的 `.wav` 文件保存到 `output/`。

## 环境

- Python 3.12
- 建议使用 NVIDIA GPU。CPU 可以尝试运行，但 1.7B 模型会很慢。
- 首次运行会下载 Qwen3-TTS 模型权重。也可以提前下载到本地，并通过 `QWEN_TTS_MODEL` 指向本地目录。

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

1. 上传一段参考音频，建议是清晰、干净的短音频。
2. 输入参考音频对应的逐字文本。
3. 输入一行或多行目标文本，每一行会生成一个独立音频文件。
4. 可选填写“情绪/语气”。这是实验性功能：Qwen3-TTS Base voice clone 官方示例没有稳定的 `instruct` 参数，本程序会用保守的文本提示方式尝试表达语气，并在 `metadata.json` 中记录。
5. 生成结果默认保存在 `output/YYYYMMDD-HHMMSS/`。

## 配置

- `QWEN_TTS_MODEL`：模型 ID 或本地模型目录，默认 `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
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

`metadata.json` 会记录参考素材、语言、实验性情绪提示和输出文件信息。

## 参考

- [Qwen3-TTS 1.7B Base 模型卡](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- [Qwen3-TTS GitHub](https://github.com/QwenLM/Qwen3-TTS)
