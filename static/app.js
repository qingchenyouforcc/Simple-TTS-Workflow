const form = document.querySelector("#tts-form");
const modeSelect = document.querySelector("#mode-select");
const modeNote = document.querySelector("#mode-note");
const submitButton = document.querySelector("#submit-button");
const message = document.querySelector("#message");
const resultList = document.querySelector("#result-list");

const modeNotes = {
  clone: "使用 Base 模型克隆上传音频。语气来自参考音频本身，不会把情绪描述拼进正文。",
  voice_design: "使用 VoiceDesign 模型，情绪/语气描述会作为 instruct 参数生效。",
  voice_design_then_clone: "先设计一段风格参考音频，再复用为 clone prompt 批量生成目标文本。",
};

modeSelect.addEventListener("change", updateModeUI);
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
  modeNote.textContent = modeNotes[mode];

  for (const group of document.querySelectorAll("[data-mode-group]")) {
    const modes = group.dataset.modeGroup.split(" ");
    const visible =
      modes.includes(mode) ||
      (group.dataset.modeGroup === "design" && mode !== "clone");
    group.hidden = !visible;
  }

  for (const field of form.querySelectorAll("[data-required-when]")) {
    const modes = field.dataset.requiredWhen.split(" ");
    field.required = modes.includes(mode);
  }
}
