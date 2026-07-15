/** 分步加载进度 UI */
import { $, esc } from "./utils.js";

const PROGRESS = { steps: [], active: -1 };

export function showLoading(on, text) {
  $("loading")?.classList.toggle("hide", !on);
  if (text && $("loading-text")) $("loading-text").textContent = text;
  if (!on) {
    PROGRESS.steps = [];
    PROGRESS.active = -1;
  }
}

export function startProgress(title, steps) {
  PROGRESS.steps = steps.map((label) => ({ label, status: "pending" }));
  PROGRESS.active = -1;
  $("loading")?.classList.remove("hide");
  if ($("loading-text")) $("loading-text").textContent = title || "分析中…";
  if ($("loading-sub")) $("loading-sub").textContent = "初始化…";
  if ($("loading-bar")) $("loading-bar").style.width = "2%";
  if ($("loading-pct")) $("loading-pct").textContent = "0%";
  renderProgressSteps();
}

function renderProgressSteps() {
  const ul = $("loading-steps");
  if (!ul) return;
  ul.innerHTML = PROGRESS.steps.map((s) => {
    let cls = "", mark = "○";
    if (s.status === "done") { cls = "done"; mark = "✓"; }
    else if (s.status === "active") { cls = "active"; mark = "●"; }
    else if (s.status === "err") { cls = "err"; mark = "!"; }
    else if (s.status === "skip") { cls = "done"; mark = "–"; }
    return `<li class="${cls}"><span><span class="mark">${mark}</span> ${esc(s.label)}</span></li>`;
  }).join("");
}

export function setProgress(stepIndex, status, subText) {
  if (stepIndex < 0 || stepIndex >= PROGRESS.steps.length) return;
  for (let i = 0; i < stepIndex; i++) {
    if (PROGRESS.steps[i].status !== "err" && PROGRESS.steps[i].status !== "skip") {
      PROGRESS.steps[i].status = "done";
    }
  }
  PROGRESS.steps[stepIndex].status = status || "active";
  PROGRESS.active = stepIndex;
  const doneCount = PROGRESS.steps.filter((s) => s.status === "done" || s.status === "skip").length;
  const activeBoost = status === "active" ? 0.45 : status === "done" ? 1 : 0;
  const pct = Math.min(99, Math.round(((doneCount + activeBoost) / PROGRESS.steps.length) * 100));
  if ($("loading-bar")) $("loading-bar").style.width = pct + "%";
  if ($("loading-pct")) $("loading-pct").textContent = pct + "%";
  if (subText && $("loading-sub")) $("loading-sub").textContent = subText;
  else if (status === "active" && $("loading-sub")) {
    $("loading-sub").textContent = PROGRESS.steps[stepIndex].label + "…";
  }
  renderProgressSteps();
}

export function finishProgress(ok, message) {
  if (ok) {
    PROGRESS.steps.forEach((s) => {
      if (s.status === "active" || s.status === "pending") s.status = "done";
    });
    if ($("loading-bar")) $("loading-bar").style.width = "100%";
    if ($("loading-pct")) $("loading-pct").textContent = "100%";
    if ($("loading-sub")) $("loading-sub").textContent = message || "完成";
    if ($("loading-text")) $("loading-text").textContent = "分析完成";
    renderProgressSteps();
    setTimeout(() => showLoading(false), 450);
  } else {
    if (PROGRESS.active >= 0 && PROGRESS.steps[PROGRESS.active]) {
      PROGRESS.steps[PROGRESS.active].status = "err";
    }
    if ($("loading-sub")) $("loading-sub").textContent = message || "失败";
    if ($("loading-text")) $("loading-text").textContent = "分析失败";
    renderProgressSteps();
    setTimeout(() => showLoading(false), 1200);
  }
}

/** 允许 analysis 模块改写步骤标签（如水系降级） */
export function relabelStep(index, label) {
  if (PROGRESS.steps[index]) PROGRESS.steps[index].label = label;
}

export function getProgress() {
  return PROGRESS;
}
