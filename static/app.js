const form = document.querySelector("#tts-form");
const submitButton = document.querySelector("#submit-button");
const message = document.querySelector("#message");
const resultList = document.querySelector("#result-list");

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

