const form = document.querySelector("#tts-form");
const modeInput = document.querySelector("#mode-select");
const modeCards = Array.from(document.querySelectorAll("[data-mode-value]"));
const modeNote = document.querySelector("#mode-note");
const modeCategory = document.querySelector("#mode-category");
const pageTitle = document.querySelector("#page-title");
const viewPanels = Array.from(document.querySelectorAll("[data-view-panel]"));
const resultsNav = document.querySelector("#results-nav");
const resultsCount = document.querySelector("#results-count");
const backToStudio = document.querySelector("#back-to-studio");
const sidebar = document.querySelector("#app-sidebar");
const sidebarOpen = document.querySelector("#sidebar-open");
const sidebarClose = document.querySelector("#sidebar-close");
const sidebarOverlay = document.querySelector("#sidebar-overlay");
const sidebarMedia = window.matchMedia("(max-width: 1100px)");
const voicePresetSelect = document.querySelector("#voice-preset-select");
const voicePresetNote = document.querySelector("#voice-preset-note");
const refAudioInput = document.querySelector("#ref-audio-input");
const uploadField = document.querySelector(".upload-field");
const fileName = document.querySelector("#file-name");
const targetTexts = document.querySelector("#target-texts");
const lineCount = document.querySelector("#line-count");
const charCount = document.querySelector("#char-count");
const readiness = document.querySelector(".generation-readiness");
const readinessText = document.querySelector("#readiness-text");
const submitButton = document.querySelector("#submit-button");
const buttonLabel = submitButton.querySelector(".button-label");
const message = document.querySelector("#message");
const emptyResults = document.querySelector("#empty-results");
const resultList = document.querySelector("#result-list");
const resultStatus = document.querySelector("#result-status");
const clearResults = document.querySelector("#clear-results");
const themeToggle = document.querySelector("#theme-toggle");
const toast = document.querySelector("#toast");
const cloneModes = ["vox_controllable_clone", "scene_dubbing", "vox_hifi_clone", "clone"];

let voicePresets = [];
let activePresetName = "";
let manualRefText = "";
let toastTimer;
let currentView = "studio";

const modeNotes = {
  vox_controllable_clone: "推荐入门使用。上传一段参考音频保留音色，再用自然语言控制情绪、节奏与表达方式。",
  scene_dubbing: "Qwen3.5 会逐行理解文本情景并自动设计情绪与表达，再由 VoxCPM2 使用指定音色完成配音。",
  vox_design: "不需要参考音频。描述你想要的声音与语气，VoxCPM2 会直接生成一个新的表达。",
  vox_hifi_clone: "适合更看重音色相似度的场景。需要参考音频及逐字文本，此模式不使用语气描述。",
  clone: "使用 Qwen3-TTS Base 忠实克隆参考音频。表达方式主要来自参考音频本身。",
  voice_design: "使用 Qwen3-TTS VoiceDesign，根据自然语言指令直接设计音色、情绪与说话方式。",
  voice_design_then_clone: "先生成一段符合描述的风格参考音频，再把它复用到多行文本，保持批量输出一致。",
};

const modeMeta = {
  vox_controllable_clone: { title: "可控克隆", category: "声音克隆 · VOXCPM2" },
  scene_dubbing: { title: "情景配音", category: "智能配音 · QWEN3.5 + VOXCPM2" },
  vox_design: { title: "语音设计", category: "声音设计 · VOXCPM2" },
  vox_hifi_clone: { title: "Hi-Fi 克隆", category: "声音克隆 · VOXCPM2" },
  clone: { title: "参考音频克隆", category: "声音克隆 · QWEN3-TTS" },
  voice_design: { title: "语气设计", category: "声音设计 · QWEN3-TTS" },
  voice_design_then_clone: { title: "设计后复用", category: "声音设计 · QWEN3-TTS" },
};

const visibility = {
  reference_audio: ["vox_controllable_clone", "scene_dubbing", "vox_hifi_clone", "clone"],
  voice_preset: cloneModes,
  reference_text: ["vox_hifi_clone", "clone"],
  style: ["vox_controllable_clone", "vox_design", "voice_design", "voice_design_then_clone"],
  voice_design_then_clone: ["voice_design_then_clone"],
  vox_params: ["vox_controllable_clone", "scene_dubbing", "vox_design", "vox_hifi_clone"],
};

initializeTheme();
bindEvents();
syncSidebarState();
loadVoicePresets();
updateModeUI();
updateTextStats();

function bindEvents() {
  for (const card of modeCards) {
    card.addEventListener("click", () => {
      modeInput.value = card.dataset.modeValue;
      updateModeUI();
      showView("studio");
      closeSidebar(true);
    });
  }

  resultsNav.addEventListener("click", () => {
    showView("results");
    closeSidebar(true);
  });
  backToStudio.addEventListener("click", () => showView("studio"));
  sidebarOpen.addEventListener("click", openSidebar);
  sidebarClose.addEventListener("click", () => closeSidebar(true));
  sidebarOverlay.addEventListener("click", () => closeSidebar(true));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && document.body.classList.contains("is-sidebar-open")) {
      closeSidebar(true);
    }
  });
  sidebarMedia.addEventListener("change", syncSidebarState);

  voicePresetSelect.addEventListener("change", applyPresetSelection);
  refAudioInput.addEventListener("change", updateUploadState);
  targetTexts.addEventListener("input", () => {
    updateTextStats();
    clearInvalidState(targetTexts);
  });

  for (const field of form.querySelectorAll("input, select, textarea")) {
    field.addEventListener("input", () => {
      clearInvalidState(field);
      updateReadiness();
    });
  }

  for (const chip of document.querySelectorAll("[data-style-prompt]")) {
    chip.addEventListener("click", () => {
      const styleField = form.elements.emotion_instruction;
      styleField.value = chip.dataset.stylePrompt;
      styleField.dispatchEvent(new Event("input", { bubbles: true }));
      styleField.focus();
    });
  }

  for (const eventName of ["dragenter", "dragover"]) {
    uploadField.addEventListener(eventName, () => uploadField.classList.add("is-dragging"));
  }
  for (const eventName of ["dragleave", "drop"]) {
    uploadField.addEventListener(eventName, () => uploadField.classList.remove("is-dragging"));
  }

  form.addEventListener("submit", handleSubmit);
  clearResults.addEventListener("click", resetResults);
  themeToggle.addEventListener("click", toggleTheme);
}

async function handleSubmit(event) {
  event.preventDefault();
  if (!validateForm()) {
    return;
  }

  setLoading(true);
  setResultState("loading", "正在生成");
  message.className = "message";
  message.textContent = modeInput.value === "scene_dubbing"
    ? "Qwen3.5 正在逐行分析情绪，随后将由 VoxCPM2 生成音频。首次使用需要下载模型，请保持页面开启。"
    : "模型正在准备并生成音频。首次使用可能还需要下载权重，请保持页面开启。";
  emptyResults.hidden = true;
  resultList.replaceChildren();
  updateResultsCount(0);
  clearResults.hidden = true;

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      body: new FormData(form),
    });
    const payload = await parseResponse(response);
    if (!response.ok) {
      throw new Error(payload.detail || "生成失败，请检查输入后重试。");
    }

    renderSuccessMessage(payload);
    const analysesByIndex = new Map(
      (payload.emotion_analyses || []).map((analysis) => [analysis.index, analysis]),
    );
    payload.items.forEach((item, index) => {
      resultList.appendChild(renderResult(item, index, analysesByIndex.get(item.index)));
    });
    updateResultsCount(payload.items.length);
    setResultState("success", "生成完成");
    clearResults.hidden = false;
    showToast("已生成 " + payload.items.length + " 个音频");
    showView("results");
  } catch (error) {
    message.className = "message is-error";
    message.textContent = error instanceof Error ? error.message : "生成失败，请稍后重试。";
    emptyResults.hidden = false;
    setResultState("error", "生成失败");
  } finally {
    setLoading(false);
  }
}

async function parseResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }
  const body = await response.text();
  return { detail: body || "服务器返回了无法识别的响应。" };
}

function renderSuccessMessage(payload) {
  message.className = "message success-message";
  const summary = document.createElement("span");
  summary.textContent = "已生成 " + payload.items.length + " 个音频 · " + payload.output_dir;

  const copyButton = document.createElement("button");
  copyButton.type = "button";
  copyButton.className = "inline-copy";
  copyButton.textContent = "复制目录";
  copyButton.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(payload.output_dir);
      showToast("输出目录已复制");
    } catch {
      showToast("无法访问剪贴板，请手动复制目录");
    }
  });
  message.replaceChildren(summary, copyButton);
}

function renderResult(item, animationIndex, emotionAnalysis) {
  const wrapper = document.createElement("article");
  wrapper.className = "result-item";
  wrapper.style.animationDelay = String(animationIndex * 55) + "ms";

  const details = document.createElement("div");
  details.className = "result-details";
  const index = document.createElement("span");
  index.className = "result-index";
  index.textContent = String(item.index).padStart(2, "0");

  const copy = document.createElement("div");
  copy.className = "result-copy";
  const title = document.createElement("strong");
  title.textContent = item.filename || "音频 " + item.index;
  const text = document.createElement("p");
  text.textContent = item.text;
  copy.append(title, text);
  if (emotionAnalysis) {
    const emotion = document.createElement("p");
    emotion.className = "result-emotion";
    const label = document.createElement("strong");
    label.textContent = "情绪分析";
    const instruction = document.createElement("span");
    instruction.textContent = emotionAnalysis.instruction;
    emotion.append(label, instruction);
    copy.append(emotion);
  }
  details.append(index, copy);

  const player = document.createElement("div");
  player.className = "result-player";
  const audio = document.createElement("audio");
  audio.controls = true;
  audio.preload = "metadata";
  audio.src = item.url;
  audio.setAttribute("aria-label", "试听 " + title.textContent);

  const download = document.createElement("a");
  download.className = "download-link";
  download.href = item.url;
  download.download = item.filename || "";
  download.title = "下载 " + title.textContent;
  download.setAttribute("aria-label", download.title);
  download.textContent = "↓";
  player.append(audio, download);

  wrapper.append(details, player);
  return wrapper;
}

function updateModeUI() {
  const mode = modeInput.value;
  const meta = modeMeta[mode];
  const selectedPreset = cloneModes.includes(mode) ? getSelectedVoicePreset() : null;
  modeNote.textContent = modeNotes[mode];
  pageTitle.textContent = meta.title;
  modeCategory.textContent = meta.category;

  for (const card of modeCards) {
    const isActive = card.dataset.modeValue === mode;
    card.classList.toggle("is-active", isActive);
    if (isActive && currentView === "studio") {
      card.setAttribute("aria-current", "page");
    } else {
      card.removeAttribute("aria-current");
    }
  }

  for (const group of document.querySelectorAll("[data-mode-group]")) {
    const modes = visibility[group.dataset.modeGroup] || group.dataset.modeGroup.split(" ");
    group.hidden = !modes.includes(mode);
  }

  for (const field of form.querySelectorAll("[data-required-when]")) {
    const modes = field.dataset.requiredWhen.split(" ");
    field.required = modes.includes(mode) && !selectedPreset;
  }

  for (const label of form.querySelectorAll("[data-required-label]")) {
    label.hidden = !label.dataset.requiredLabel.split(" ").includes(mode);
  }

  voicePresetSelect.disabled = !cloneModes.includes(mode) || voicePresets.length === 0;
  updatePresetNote(selectedPreset);

  const styleField = form.elements.emotion_instruction;
  styleField.disabled = mode === "vox_hifi_clone";
  updateUploadState();
  updateReadiness();
}

function applyPresetSelection() {
  const selectedPreset = getSelectedVoicePreset();
  const refTextField = form.elements.ref_text;

  if (selectedPreset) {
    if (!activePresetName) {
      manualRefText = refTextField.value;
    }
    refTextField.value = selectedPreset.ref_text;
    activePresetName = selectedPreset.name;
  } else if (activePresetName) {
    refTextField.value = manualRefText;
    activePresetName = "";
  }

  updateModeUI();
}

function updatePresetNote(selectedPreset) {
  if (voicePresets.length === 0) {
    voicePresetNote.textContent = "role 文件夹中暂无可用预设，请上传参考音频。";
  } else if (selectedPreset) {
    voicePresetNote.textContent = "正在使用预设音频：" + selectedPreset.audio_filename;
  } else {
    voicePresetNote.textContent = "可直接使用 role 文件夹中的声音，也可以上传临时参考音频。";
  }
}

function updateUploadState() {
  const selectedPreset = cloneModes.includes(modeInput.value) ? getSelectedVoicePreset() : null;
  refAudioInput.disabled = Boolean(selectedPreset);
  uploadField.classList.toggle("has-file", Boolean(selectedPreset) || refAudioInput.files.length > 0);
  uploadField.classList.toggle("is-disabled", Boolean(selectedPreset));

  if (selectedPreset) {
    fileName.textContent = "已使用预设：" + selectedPreset.audio_filename;
  } else if (refAudioInput.files.length > 0) {
    const file = refAudioInput.files[0];
    fileName.textContent = file.name + " · " + formatFileSize(file.size);
  } else {
    fileName.textContent = "支持 WAV、MP3、FLAC、M4A、OGG";
  }
}

function updateTextStats() {
  const value = targetTexts.value;
  const lines = value.split(String.fromCharCode(10)).filter((line) => line.trim()).length;
  const characters = Array.from(value).filter((character) => character.trim()).length;
  lineCount.textContent = lines + " 行";
  charCount.textContent = characters + " 字";
  updateReadiness();
}

function updateReadiness() {
  const lineTotal = targetTexts.value
    .split(String.fromCharCode(10))
    .filter((line) => line.trim()).length;
  const missingRequired = Array.from(form.querySelectorAll("[required]")).some((field) => {
    if (field.disabled || field.closest("[hidden]")) {
      return false;
    }
    if (field.type === "file") {
      return field.files.length === 0;
    }
    return !field.value.trim();
  });
  const isReady = lineTotal > 0 && !missingRequired;
  readiness.classList.toggle("is-ready", isReady);
  readinessText.textContent = isReady
    ? "设置完成，将生成 " + lineTotal + " 个音频"
    : "补全必填内容后即可生成";
}

function validateForm() {
  clearAllInvalidStates();
  const requiredFields = Array.from(form.querySelectorAll("[required]"));
  for (const field of requiredFields) {
    if (field.disabled || field.closest("[hidden]")) {
      continue;
    }
    const isEmpty = field.type === "file" ? field.files.length === 0 : !field.value.trim();
    if (isEmpty) {
      markInvalid(field);
      const label = field.name === "texts" ? "目标文本" : field.name === "ref_audio" ? "参考音频" : "必填内容";
      showToast("请填写或选择" + label);
      field.focus();
      return false;
    }
  }

  for (const field of form.querySelectorAll("input[type='number']")) {
    if (!field.checkValidity()) {
      markInvalid(field);
      showToast("生成参数超出有效范围");
      field.focus();
      return false;
    }
  }
  return true;
}

function markInvalid(field) {
  const container = field.closest(".field, .upload-field, .textarea-wrap");
  if (container) {
    container.classList.add("is-invalid");
  }
  field.setAttribute("aria-invalid", "true");
}

function clearInvalidState(field) {
  const container = field.closest(".field, .upload-field, .textarea-wrap");
  if (container) {
    container.classList.remove("is-invalid");
  }
  field.removeAttribute("aria-invalid");
}

function clearAllInvalidStates() {
  for (const field of form.querySelectorAll("[aria-invalid='true']")) {
    clearInvalidState(field);
  }
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.classList.toggle("is-loading", isLoading);
  buttonLabel.textContent = isLoading
    ? modeInput.value === "scene_dubbing" ? "正在分析并生成" : "正在生成，请稍候"
    : "开始生成";
}

function setResultState(state, label) {
  resultStatus.className = "result-status is-" + state;
  resultStatus.innerHTML = "<i></i>" + label;
}

function resetResults() {
  resultList.replaceChildren();
  emptyResults.hidden = false;
  clearResults.hidden = true;
  updateResultsCount(0);
  message.className = "message";
  message.textContent = "完成设置并开始生成，你的音频会出现在这里。";
  setResultState("idle", "等待生成");
}

async function loadVoicePresets() {
  try {
    const response = await fetch("/api/voice-presets");
    const payload = await parseResponse(response);
    if (!response.ok) {
      throw new Error(payload.detail || "无法加载语音预设");
    }
    voicePresets = payload.presets || [];
    renderVoicePresetOptions();
  } catch (error) {
    voicePresets = [];
    voicePresetNote.textContent = error instanceof Error ? error.message : "无法加载语音预设";
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

function formatFileSize(size) {
  if (size < 1024 * 1024) {
    return Math.max(1, Math.round(size / 1024)) + " KB";
  }
  return (size / (1024 * 1024)).toFixed(1) + " MB";
}

function initializeTheme() {
  let storedTheme = "";
  try {
    storedTheme = localStorage.getItem("tts-theme") || "";
  } catch {
    storedTheme = "";
  }
  const systemDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  applyTheme(storedTheme || (systemDark ? "dark" : "light"));
}

function toggleTheme() {
  const current = document.documentElement.dataset.theme || "light";
  const next = current === "dark" ? "light" : "dark";
  applyTheme(next);
  try {
    localStorage.setItem("tts-theme", next);
  } catch {
    // Theme persistence is optional.
  }
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const icon = themeToggle.querySelector(".theme-icon");
  const label = themeToggle.querySelector(".theme-label");
  icon.textContent = theme === "dark" ? "☀" : "◐";
  if (label) {
    label.textContent = theme === "dark" ? "切换浅色主题" : "切换深色主题";
  }
  themeToggle.setAttribute("aria-label", theme === "dark" ? "切换到浅色主题" : "切换到深色主题");
}

function showView(view) {
  currentView = view;
  for (const panel of viewPanels) {
    panel.hidden = panel.dataset.viewPanel !== view;
  }

  const showingResults = view === "results";
  resultsNav.classList.toggle("is-active", showingResults);
  if (showingResults) {
    resultsNav.setAttribute("aria-current", "page");
  } else {
    resultsNav.removeAttribute("aria-current");
  }

  for (const card of modeCards) {
    const isCurrentMode = card.dataset.modeValue === modeInput.value && !showingResults;
    card.classList.toggle("is-active", isCurrentMode);
    if (isCurrentMode) {
      card.setAttribute("aria-current", "page");
    } else {
      card.removeAttribute("aria-current");
    }
  }

  const behavior = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
  window.scrollTo({ top: 0, behavior });
}

function updateResultsCount(count) {
  resultsCount.textContent = String(count);
  resultsCount.hidden = count === 0;
}

function openSidebar() {
  document.body.classList.add("is-sidebar-open");
  sidebarOpen.setAttribute("aria-expanded", "true");
  sidebar.setAttribute("aria-hidden", "false");
  sidebar.inert = false;
  const activeItem = sidebar.querySelector(".nav-item.is-active") || sidebarClose;
  activeItem.focus();
}

function closeSidebar(restoreFocus = false) {
  document.body.classList.remove("is-sidebar-open");
  sidebarOpen.setAttribute("aria-expanded", "false");
  if (sidebarMedia.matches) {
    sidebar.setAttribute("aria-hidden", "true");
    sidebar.inert = true;
  } else {
    sidebar.removeAttribute("aria-hidden");
    sidebar.inert = false;
  }
  if (restoreFocus && sidebarMedia.matches) {
    sidebarOpen.focus();
  }
}

function syncSidebarState() {
  closeSidebar(false);
}

function showToast(text) {
  clearTimeout(toastTimer);
  toast.textContent = text;
  toast.classList.add("is-visible");
  toastTimer = setTimeout(() => toast.classList.remove("is-visible"), 2600);
}
