const form = document.querySelector("#tts-form");
const modeSelect = document.querySelector("#mode-select");
const modeNote = document.querySelector("#mode-note");
const voicePresetSelect = document.querySelector("#voice-preset-select");
const voicePresetNote = document.querySelector("#voice-preset-note");
const submitButton = document.querySelector("#submit-button");
const message = document.querySelector("#message");
const resultList = document.querySelector("#result-list");
const cloneModes = ["vox_controllable_clone", "vox_hifi_clone", "clone"];
let voicePresets = [];

const modeNotes = {
  vox_controllable_clone: "VoxCPM2 默认模式：上传参考音频克隆音色，情绪/语气描述会控制风格。",
  vox_design: "VoxCPM2 语音设计：不需要参考音频，用语气描述直接生成新声音。",
  vox_hifi_clone: "VoxCPM2 Hi-Fi 克隆：需要参考文本以提高相似度；此模式会忽略语气描述。",
  clone: "Qwen3-TTS Base：克隆上传音频。语气来自参考音频本身，不支持独立语气控制。",
  voice_design: "Qwen3-TTS VoiceDesign：情绪/语气描述会作为 instruct 参数生效。",
  voice_design_then_clone: "Qwen3-TTS：先设计风格参考音频，再复用为 clone prompt 批量生成目标文本。",
};

const visibility = {
  reference_audio: ["vox_controllable_clone", "vox_hifi_clone", "clone"],
  voice_preset: cloneModes,
  reference_text: ["vox_hifi_clone", "clone"],
  style: ["vox_controllable_clone", "vox_design", "voice_design", "voice_design_then_clone"],
  voice_design_then_clone: ["voice_design_then_clone"],
  vox_params: ["vox_controllable_clone", "vox_design", "vox_hifi_clone"],
};

modeSelect.addEventListener("change", updateModeUI);
voicePresetSelect.addEventListener("change", updateModeUI);
loadVoicePresets();
updateModeUI();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  submitButton.textContent = "生成中...";
  message.textContent = "模型生成可能需要一些时间，请稍等。";
  resultList.replaceChildren();

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      body: new FormData(form),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "生成失败");
    }

    message.textContent = `已生成 ${payload.items.length} 个音频，输出目录：${payload.output_dir}`;
    for (const item of payload.items) {
      resultList.appendChild(renderResult(item));
    }
  } catch (error) {
    message.textContent = error.message;
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "开始生成";
  }
});

function renderResult(item) {
  const wrapper = document.createElement("article");
  wrapper.className = "result-item";

  const details = document.createElement("div");
  const title = document.createElement("strong");
  title.textContent = `Line ${String(item.index).padStart(3, "0")}`;
  const text = document.createElement("p");
  text.textContent = item.text;
  const path = document.createElement("code");
  path.textContent = item.path;
  details.append(title, text, path);

  const audio = document.createElement("audio");
  audio.controls = true;
  audio.src = item.url;

  wrapper.append(details, audio);
  return wrapper;
}

function updateModeUI() {
  const mode = modeSelect.value;
  const selectedPreset = cloneModes.includes(mode) ? getSelectedVoicePreset() : null;
  modeNote.textContent = modeNotes[mode];

  for (const group of document.querySelectorAll("[data-mode-group]")) {
    const modes = visibility[group.dataset.modeGroup] || group.dataset.modeGroup.split(" ");
    group.hidden = !modes.includes(mode);
  }

  for (const field of form.querySelectorAll("[data-required-when]")) {
    const modes = field.dataset.requiredWhen.split(" ");
    field.required = modes.includes(mode) && !selectedPreset;
  }

  voicePresetSelect.disabled = !cloneModes.includes(mode) || voicePresets.length === 0;
  if (voicePresets.length === 0) {
    voicePresetNote.textContent = "role 文件夹中没有可用预设，将使用上传参考素材。";
  } else if (selectedPreset) {
    voicePresetNote.textContent = `使用预设音频：${selectedPreset.audio_filename}`;
    if (form.elements.ref_text && cloneModes.includes(mode)) {
      form.elements.ref_text.value = selectedPreset.ref_text;
    }
  } else {
    voicePresetNote.textContent = "可选择 role 文件夹中的预设，或继续上传参考音频。";
  }

  const styleField = form.elements.emotion_instruction;
  styleField.disabled = mode === "vox_hifi_clone";
  if (styleField.disabled) {
    styleField.value = "";
  }
}

async function loadVoicePresets() {
  try {
    const response = await fetch("/api/voice-presets");
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "无法加载语音预设");
    }
    voicePresets = payload.presets || [];
    renderVoicePresetOptions();
  } catch (error) {
    voicePresets = [];
    voicePresetNote.textContent = error.message;
  } finally {
    updateModeUI();
  }
}

function renderVoicePresetOptions() {
  const currentValue = voicePresetSelect.value;
  voicePresetSelect.replaceChildren(new Option("不使用预设", ""));
  for (const preset of voicePresets) {
    voicePresetSelect.appendChild(new Option(preset.name, preset.name));
  }
  if (voicePresets.some((preset) => preset.name === currentValue)) {
    voicePresetSelect.value = currentValue;
  }
}

function getSelectedVoicePreset() {
  const selectedName = voicePresetSelect.value;
  return voicePresets.find((preset) => preset.name === selectedName) || null;
}
