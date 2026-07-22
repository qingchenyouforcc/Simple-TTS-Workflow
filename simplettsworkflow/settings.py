from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR = BASE_DIR / ".tmp_uploads"
ROLE_DIR = BASE_DIR / "role"
MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
VOICE_DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
VOXCPM_MODEL_ID = "openbmb/VoxCPM2"
EMOTION_MODEL_REPO = "bartowski/Qwen_Qwen3.5-2B-GGUF"
EMOTION_MODEL_FILE = "Qwen_Qwen3.5-2B-Q4_K_M.gguf"
