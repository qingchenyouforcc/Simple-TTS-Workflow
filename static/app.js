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
const targetLabelText = document.querySelector("#target-label-text");
const lineCount = document.querySelector("#line-count");
const charCount = document.querySelector("#char-count");
const sceneDubbingModeInput = document.querySelector("#scene-dubbing-mode");
const sceneDubbingModeButtons = Array.from(document.querySelectorAll("[data-scene-dubbing-mode]"));
const sceneModeHelp = document.querySelector("#scene-mode-help");
const batchTipText = document.querySelector("#batch-tip-text");
const readiness = document.querySelector(".generation-readiness");
const readinessText = document.querySelector("#readiness-text");
const submitButton = document.querySelector("#submit-button");
const buttonLabel = submitButton.querySelector(".button-label");
const message = document.querySelector("#message");
const emptyResults = document.querySelector("#empty-results");
const resultList = document.querySelector("#result-list");
const resultStatus = document.querySelector("#result-status");
const clearResults = document.querySelector("#clear-results");
const regenerateButton = document.querySelector("#regenerate-button");
const regenerateLabel = regenerateButton.querySelector(".regenerate-label");
const retainCandidates = document.querySelector("#retain-candidates");
const candidateHelp = document.querySelector("#candidate-help");
const themeToggle = document.querySelector("#theme-toggle");
const toast = document.querySelector("#toast");
const cloneModes = ["vox_controllable_clone", "scene_dubbing", "vox_hifi_clone", "clone"];

let voicePresets = [];
let activePresetName = "";
let manualRefText = "";
let toastTimer;
let currentView = "studio";
let lastGenerationData = null;
let generationInProgress = false;
let resultGroups = [];
let regenerationCount = loadRegenerationCount();

function loadRegenerationCount() {
  try {
    const value = Number(localStorage.getItem("tts-regeneration-count"));
    return Number.isInteger(value) && value >= 1 && value <= 20 ? value : 3;
  } catch { return 3; }
}

const modeNotes = {
  vox_controllable_clone: "推荐入门使用。上传一段参考音频保留音色，再用自然语言控制情绪、节奏与表达方式。",
  scene_dubbing: "Qwen3.5 会理解文本情景并设计情绪与表达，再由 VoxCPM2 使用指定音色完成配音。",
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
  let settingsReturnView = "studio";
  const countInput = document.querySelector("#regeneration-count");
  document.querySelector("#open-settings").addEventListener("click", () => {
    countInput.value = String(regenerationCount);
    if (currentView !== "settings") settingsReturnView = currentView;
    showView("settings");
    closeSidebar(true);
  });
  document.querySelector("#close-settings").addEventListener("click", () => showView(settingsReturnView));
  document.querySelector("#settings-form").addEventListener("submit", (event) => {
    event.preventDefault();
    if (!countInput.reportValidity()) return;
    regenerationCount = Number(countInput.value);
    let persisted = true;
    try { localStorage.setItem("tts-regeneration-count", String(regenerationCount)); }
    catch { persisted = false; }
    for (const button of resultList.querySelectorAll(".regenerate-one")) {
      button.textContent = "↻ 重新生成 " + regenerationCount + " 次";
    }
    showToast(persisted ? "设置已保存" : "设置已应用，浏览器未允许永久保存");
  });
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

  for (const button of sceneDubbingModeButtons) {
    button.addEventListener("click", () => {
      sceneDubbingModeInput.value = button.dataset.sceneDubbingMode;
      updateSceneDubbingUI();
      updateTextStats();
      clearInvalidState(targetTexts);
    });
  }

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
  regenerateButton.addEventListener("click", () => {
    if (lastGenerationData && !generationInProgress) {
      generateAudio(lastGenerationData);
    }
  });
  clearResults.addEventListener("click", resetResults);
  retainCandidates.addEventListener("click", retainSelectedCandidates);
  themeToggle.addEventListener("click", toggleTheme);
}

async function handleSubmit(event) {
  event.preventDefault();
  if (generationInProgress) {
    return;
  }
  if (!validateForm()) {
    return;
  }

  await generateAudio(new FormData(form));
}

async function generateAudio(data) {
  if (generationInProgress) {
    return;
  }
  const appending = data === lastGenerationData && resultGroups.length > 0;
  regenerateButton.hidden = false;
  setLoading(true);
  setResultState("loading", "正在生成");
  message.className = "message";
  if (data.get("mode") === "scene_dubbing") {
    message.textContent = data.get("scene_dubbing_mode") === "assisted"
      ? "Qwen3.5 正在根据描述与关键词生成情绪指令，随后将由 VoxCPM2 完成配音。首次使用需要下载模型，请保持页面开启。"
      : "Qwen3.5 正在自识别每行文本的情绪，随后将由 VoxCPM2 生成音频。首次使用需要下载模型，请保持页面开启。";
  } else {
    message.textContent = "模型正在准备并生成音频。首次使用可能还需要下载权重，请保持页面开启。";
  }
  emptyResults.hidden = true;
  clearResults.hidden = true;

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      body: data,
    });
    const payload = await parseResponse(response);
    if (!response.ok) {
      throw new Error(payload.detail || "生成失败，请检查输入后重试。");
    }

    const analysesByIndex = new Map(
      (payload.emotion_analyses || []).map((analysis) => [analysis.index, analysis]),
    );
    if (appending && payload.items.length !== resultGroups.length) {
      throw new Error("返回的音频条数与本次结果不一致，请重试。");
    }
    if (!appending) {
      resultGroups = payload.items.map((item, index) => ({
        index: item.index, input: getSingleInput(data, index), versions: [], selected: 0,
      }));
    }
    payload.items.forEach((item, index) => {
      const group = resultGroups[index];
      group.versions.push({ item, analysis: analysesByIndex.get(item.index), outputDir: payload.output_dir, number: (group.versions.at(-1)?.number || 0) + 1 });
    });
    lastGenerationData = data;
    renderResultGroups();
    renderCandidateSummary();
    setResultState("success", "生成完成");
    clearResults.hidden = false;
    showToast("已生成 " + payload.items.length + " 个音频");
    showView("results");
  } catch (error) {
    message.className = "message is-error";
    message.textContent = error instanceof Error ? error.message : "生成失败，请稍后重试。";
    emptyResults.hidden = resultGroups.length > 0;
    setResultState("error", "生成失败");
  } finally {
    regenerateButton.hidden = !lastGenerationData;
    clearResults.hidden = resultGroups.length === 0;
    setLoading(false);
  }
}

function getSingleInput(data, index) {
  const texts = String(data.get("texts") || "").replace(/\r\n?/g, "\n").trim();
  if (data.get("mode") === "scene_dubbing" && data.get("scene_dubbing_mode") === "assisted") {
    return texts.split(/\n(?:[ \t]*\n)+/)[index];
  }
  return texts.split("\n").map((line) => line.trim()).filter(Boolean)[index];
}

function renderResultGroups() {
  for (const audio of resultList.querySelectorAll("audio")) audio.pause();
  resultList.replaceChildren();
  for (const group of resultGroups) {
    const version = group.versions[group.selected];
    const card = renderResult({ ...version.item, index: group.index }, 0, version.analysis);
    const controls = document.createElement("div");
    controls.className = "candidate-controls";
    const choices = document.createElement("fieldset");
    choices.className = "candidate-choices";
    const legend = document.createElement("legend");
    legend.textContent = "第 " + group.index + " 条 · 选择保留的版本（切换后试听）";
    choices.append(legend);
    group.versions.forEach((candidate, index) => {
      const label = document.createElement("label");
      label.className = "candidate-choice";
      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "candidate-" + group.index;
      radio.value = String(index);
      radio.checked = group.selected === index;
      radio.disabled = generationInProgress;
      radio.addEventListener("change", () => {
        group.selected = index;
        renderResultGroups();
        renderCandidateSummary();
        resultList.querySelector(`input[name="candidate-${group.index}"]:checked`)?.focus();
      });
      label.append(radio, document.createTextNode("版本 " + candidate.number));
      choices.append(label);
    });
    const retry = document.createElement("button");
    retry.type = "button";
    retry.className = "secondary-button regenerate-one";
    retry.textContent = "↻ 重新生成 " + regenerationCount + " 次";
    retry.setAttribute("aria-label", "重新生成第 " + group.index + " 条音频");
    retry.disabled = generationInProgress;
    retry.addEventListener("click", () => regenerateOne(group));
    controls.append(choices, retry);
    card.append(controls);
    resultList.append(card);
  }
  updateResultsCount(resultGroups.length);
  candidateHelp.hidden = resultGroups.length === 0;
  retainCandidates.hidden = !resultGroups.some((group) => group.versions.length > 1);
}

function renderCandidateSummary() {
  const total = resultGroups.reduce((count, group) => count + group.versions.length, 0);
  message.className = "message";
  message.textContent = resultGroups.length + " 条音频 · " + total + " 个候选版本 · 每条保留一个选中版本";
  const directories = [...new Set(resultGroups.map((group) => group.versions[group.selected].outputDir))];
  for (const directory of directories) {
    const row = document.createElement("div");
    row.className = "candidate-directory";
    const path = document.createElement("span");
    path.textContent = directory;
    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "inline-copy";
    copy.textContent = "复制目录";
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(directory); showToast("输出目录已复制"); }
      catch { showToast("无法访问剪贴板，请手动复制目录"); }
    });
    row.append(path, copy);
    message.append(row);
  }
}

async function regenerateOne(group) {
  if (generationInProgress || !lastGenerationData) return;
  const data = new FormData();
  for (const [key, value] of lastGenerationData.entries()) data.append(key, value);
  data.set("texts", group.input);
  const count = regenerationCount;
  let completed = 0;
  setLoading(true);
  try {
    for (let attempt = 1; attempt <= count; attempt += 1) {
    setResultState("loading", "第 " + group.index + " 条 · 正在生成 " + attempt + "/" + count);
    const response = await fetch("/api/generate", { method: "POST", body: data });
    const payload = await parseResponse(response);
    if (!response.ok) throw new Error(payload.detail || "重新生成失败，请重试。");
    if (payload.items?.length !== 1) throw new Error("服务器未返回单条音频，请重试。");
    group.versions.push({ item: payload.items[0], analysis: payload.emotion_analyses?.[0], outputDir: payload.output_dir, number: group.versions.at(-1).number + 1 });
    completed += 1;
    renderResultGroups();
    renderCandidateSummary();
    }
    setResultState("success", "新版本已生成");
    showToast("第 " + group.index + " 条已新增 " + completed + " 个版本，可切换试听后选择");
  } catch (error) {
    setResultState("error", "重新生成失败");
    showToast("已生成 " + completed + "/" + count + " 个版本，已有版本已保留。" + (error.message || "请稍后重试"));
  } finally {
    setLoading(false);
  }
}

async function retainSelectedCandidates() {
  if (generationInProgress) return;
  const groups = resultGroups.filter((group) => group.versions.length > 1).map((group) => ({
    selected: group.versions[group.selected].item.candidate_id,
    discard: group.versions.filter((_, index) => index !== group.selected).map((version) => version.item.candidate_id),
  }));
  if (!groups.length) return;
  if (groups.some((group) => !group.selected || group.discard.some((id) => !id))) {
    showToast("部分版本不支持文件清理，请重新生成后重试");
    return;
  }
  setLoading(true);
  setResultState("loading", "正在保留选中版本");
  try {
    const response = await fetch("/api/results/retain", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ groups }),
    });
    const payload = await parseResponse(response);
    if (!response.ok) throw new Error(payload.detail || "清理失败，请重试。");
    for (const group of resultGroups) {
      group.versions = [group.versions[group.selected]];
      group.selected = 0;
    }
    renderResultGroups();
    renderCandidateSummary();
    setResultState("success", "已保留选中版本");
    showToast("已删除 " + payload.removed + " 个未选中的候选音频");
  } catch (error) {
    setResultState("error", "清理失败");
    showToast(error.message || "清理失败，请重试");
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
    if (emotionAnalysis.description || (emotionAnalysis.keywords || []).length > 0) {
      const guidance = document.createElement("p");
      guidance.className = "result-emotion result-guidance";
      const guidanceLabel = document.createElement("strong");
      guidanceLabel.textContent = "辅助提示";
      const guidanceText = document.createElement("span");
      const guidanceParts = [];
      if (emotionAnalysis.description) {
        guidanceParts.push(emotionAnalysis.description);
      }
      if ((emotionAnalysis.keywords || []).length > 0) {
        guidanceParts.push("关键词：" + emotionAnalysis.keywords.join("、"));
      }
      guidanceText.textContent = guidanceParts.join(" · ");
      guidance.append(guidanceLabel, guidanceText);
      copy.append(guidance);
    }
    const emotion = document.createElement("p");
    emotion.className = "result-emotion";
    const label = document.createElement("strong");
    label.textContent = emotionAnalysis.description ? "生成指令" : "情绪分析";
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
  updateSceneDubbingUI();
  updateUploadState();
  updateTextStats();
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

function isAssistedSceneMode() {
  return modeInput.value === "scene_dubbing" && sceneDubbingModeInput.value === "assisted";
}

function updateSceneDubbingUI() {
  const assisted = isAssistedSceneMode();
  for (const button of sceneDubbingModeButtons) {
    const active = button.dataset.sceneDubbingMode === sceneDubbingModeInput.value;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  }

  if (modeInput.value !== "scene_dubbing") {
    targetLabelText.textContent = "目标文本";
    targetTexts.placeholder = "在这里输入要生成的内容…\n\n每一行会生成一个独立音频。";
    replaceBatchTip("批量生成", "每个非空行会输出一个 WAV 文件。");
    return;
  }

  if (assisted) {
    modeNote.textContent = "为每条原句补充情感描述和关键词，Qwen3.5 会据此生成 VoxCPM2 所需的情感指令。";
    targetLabelText.textContent = "原句与情绪提示";
    targetTexts.placeholder = [
      "Alright, I should probably fix the filter first…",
      "体现疲惫与无奈，但保持稳定，不破音。",
      "keyword：疲惫，无奈，认命，倦怠，无力，勉强接受",
      "",
      "多条内容之间请留一个空行。",
    ].join("\n");
    sceneModeHelp.textContent = "每个条目严格填写三行：原句、情感描述、keyword：关键词；多个条目之间用空行分隔。";
    replaceBatchTip("三行一组", "每个有效条目会输出一个 WAV 文件。");
  } else {
    modeNote.textContent = "Qwen3.5 会逐行理解文本情景并自动设计情绪与表达，再由 VoxCPM2 使用指定音色完成配音。";
    targetLabelText.textContent = "目标文本";
    targetTexts.placeholder = "在这里输入要生成的内容…\n\n每一行会生成一个独立音频。";
    sceneModeHelp.textContent = "每个非空行作为一条原句，由 Qwen 自动识别情绪和表达方式。";
    replaceBatchTip("批量生成", "每个非空行会输出一个 WAV 文件。");
  }
}

function replaceBatchTip(title, description) {
  const strong = document.createElement("strong");
  strong.textContent = title;
  batchTipText.replaceChildren(strong, document.createTextNode(description));
}

function parseAssistedSceneBlocks(value) {
  const normalized = value.replace(/\r\n?/g, "\n").trim();
  if (!normalized) {
    return { items: [], blockCount: 0, error: "请至少填写一个辅助配音条目。" };
  }

  const blocks = normalized.split(/\n(?:[ \t]*\n)+/);
  const items = [];
  for (let index = 0; index < blocks.length; index += 1) {
    const lines = blocks[index].split("\n").map((line) => line.trim()).filter(Boolean);
    if (lines.length !== 3) {
      return {
        items,
        blockCount: blocks.length,
        error: "第 " + (index + 1) + " 个辅助条目必须包含原句、情感描述和 keyword 三行，当前为 " + lines.length + " 行。",
      };
    }
    const keywordMatch = lines[2].match(/^keyword\s*[:：]\s*(.*)$/i);
    if (!keywordMatch) {
      return {
        items,
        blockCount: blocks.length,
        error: "第 " + (index + 1) + " 个辅助条目的第三行必须使用 keyword：关键词 格式。",
      };
    }
    const keywords = keywordMatch[1].split(/[,，、]/).map((keyword) => keyword.trim()).filter(Boolean);
    if (keywords.length === 0) {
      return {
        items,
        blockCount: blocks.length,
        error: "第 " + (index + 1) + " 个辅助条目的 keyword 内容不能为空。",
      };
    }
    items.push({
      text: lines[0],
      description: lines[1],
      keywords,
    });
  }
  return { items, blockCount: blocks.length, error: "" };
}

function getTargetEntryState() {
  if (isAssistedSceneMode()) {
    const parsed = parseAssistedSceneBlocks(targetTexts.value);
    return {
      count: parsed.blockCount,
      valid: !parsed.error,
      error: parsed.error,
    };
  }
  const count = targetTexts.value
    .split(String.fromCharCode(10))
    .filter((line) => line.trim()).length;
  return { count, valid: count > 0, error: "" };
}

function updateTextStats() {
  const value = targetTexts.value;
  const entryState = getTargetEntryState();
  const characters = Array.from(value).filter((character) => character.trim()).length;
  lineCount.textContent = entryState.count + (isAssistedSceneMode() ? " 条" : " 行");
  charCount.textContent = characters + " 字";
  updateReadiness();
}

function updateReadiness() {
  const entryState = getTargetEntryState();
  const missingRequired = Array.from(form.querySelectorAll("[required]")).some((field) => {
    if (field.disabled || field.closest("[hidden]")) {
      return false;
    }
    if (field.type === "file") {
      return field.files.length === 0;
    }
    return !field.value.trim();
  });
  const isReady = entryState.count > 0 && entryState.valid && !missingRequired;
  readiness.classList.toggle("is-ready", isReady);
  readinessText.textContent = isReady
    ? "设置完成，将生成 " + entryState.count + " 个音频"
    : entryState.count > 0 && entryState.error
      ? entryState.error
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
  if (isAssistedSceneMode()) {
    const parsed = parseAssistedSceneBlocks(targetTexts.value);
    if (parsed.error) {
      markInvalid(targetTexts);
      showToast(parsed.error);
      targetTexts.focus();
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
  generationInProgress = isLoading;
  clearResults.disabled = isLoading;
  retainCandidates.disabled = isLoading;
  for (const control of resultList.querySelectorAll("button, input")) control.disabled = isLoading;
  regenerateButton.disabled = isLoading;
  regenerateButton.setAttribute("aria-busy", String(isLoading));
  regenerateLabel.textContent = isLoading ? "正在生成…" : "重新生成";
  submitButton.disabled = isLoading;
  submitButton.classList.toggle("is-loading", isLoading);
  if (!isLoading) {
    buttonLabel.textContent = "开始生成";
  } else if (modeInput.value !== "scene_dubbing") {
    buttonLabel.textContent = "正在生成，请稍候";
  } else {
    buttonLabel.textContent = isAssistedSceneMode() ? "正在生成指令并配音" : "正在自识别并生成";
  }
}

function setResultState(state, label) {
  resultStatus.className = "result-status is-" + state;
  resultStatus.innerHTML = "<i></i>" + label;
}

function resetResults() {
  if (generationInProgress) {
    return;
  }
  lastGenerationData = null;
  resultGroups = [];
  retainCandidates.hidden = true;
  candidateHelp.hidden = true;
  regenerateButton.hidden = true;
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
  const settingsNav = document.querySelector("#open-settings");
  settingsNav.classList.toggle("is-active", view === "settings");
  if (view === "settings") settingsNav.setAttribute("aria-current", "page");
  else settingsNav.removeAttribute("aria-current");
  resultsNav.classList.toggle("is-active", showingResults);
  if (showingResults) {
    resultsNav.setAttribute("aria-current", "page");
  } else {
    resultsNav.removeAttribute("aria-current");
  }

  for (const card of modeCards) {
    const isCurrentMode = card.dataset.modeValue === modeInput.value && view === "studio";
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
