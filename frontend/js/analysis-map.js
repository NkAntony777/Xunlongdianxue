/** 分析结果画布：平移 / 缩放 */
import { state } from "./state.js";
import { $ } from "./utils.js";

export const view = {
  baseW: 0,
  baseH: 0,
  scale: 1,
  tx: 0,
  ty: 0,
  minScale: 0.4,
  maxScale: 12,
};

let drag = null;
let hintTimer = null;
let wired = false;

export function applyView() {
  const stack = $("layer-stack");
  if (!stack) return;
  stack.style.width = Math.max(1, Math.round(view.baseW)) + "px";
  stack.style.height = Math.max(1, Math.round(view.baseH)) + "px";
  stack.style.transform = `translate(${view.tx}px, ${view.ty}px) scale(${view.scale})`;
  const pct = Math.round(view.scale * 100);
  const zl = $("zoom-level");
  if (zl) zl.textContent = pct + "%";
}

export function fitLayerStack(resetView = false) {
  const stack = $("layer-stack");
  const host = $("stage-canvas");
  if (!state.dem || !host || !stack) return;
  const ar = (state.dem.width || 1) / (state.dem.height || 1);
  const availW = Math.max(40, host.clientWidth - 16);
  const availH = Math.max(40, host.clientHeight - 16);
  let w = availW, h = w / ar;
  if (h > availH) { h = availH; w = h * ar; }
  view.baseW = Math.floor(w);
  view.baseH = Math.floor(h);
  if (resetView || view.scale === 0) {
    view.scale = 1;
    view.tx = (host.clientWidth - view.baseW) / 2;
    view.ty = (host.clientHeight - view.baseH) / 2;
  } else if (!Number.isFinite(view.tx) || !Number.isFinite(view.ty)) {
    view.tx = (host.clientWidth - view.baseW * view.scale) / 2;
    view.ty = (host.clientHeight - view.baseH * view.scale) / 2;
  }
  applyView();
}

export function zoomAt(clientX, clientY, factor) {
  const host = $("stage-canvas");
  if (!host || !view.baseW) return;
  const rect = host.getBoundingClientRect();
  const mx = clientX - rect.left;
  const my = clientY - rect.top;
  const prev = view.scale;
  let next = Math.min(view.maxScale, Math.max(view.minScale, prev * factor));
  if (next === prev) return;
  const cx = (mx - view.tx) / prev;
  const cy = (my - view.ty) / prev;
  view.scale = next;
  view.tx = mx - cx * next;
  view.ty = my - cy * next;
  applyView();
}

export function zoomBy(factor) {
  const host = $("stage-canvas");
  if (!host) return;
  const r = host.getBoundingClientRect();
  zoomAt(r.left + r.width / 2, r.top + r.height / 2, factor);
}

export function resetView() {
  fitLayerStack(true);
}

export function showZoomHint() {
  const el = $("zoom-hint");
  if (!el) return;
  el.classList.add("show");
  clearTimeout(hintTimer);
  hintTimer = setTimeout(() => el.classList.remove("show"), 3200);
}

export function onAnalysisViewShown() {
  if (state.dem) {
    fitLayerStack(false);
    applyView();
  }
}

export function wireAnalysisMap() {
  if (wired) return;
  wired = true;
  const host = $("stage-canvas");
  if (!host) return;
  host.tabIndex = 0;

  host.addEventListener("wheel", (e) => {
    e.preventDefault();
    zoomAt(e.clientX, e.clientY, e.deltaY < 0 ? 1.12 : 1 / 1.12);
  }, { passive: false });

  host.addEventListener("pointerdown", (e) => {
    if (e.button !== 0 && e.pointerType === "mouse") return;
    if (e.target.closest("#zoom-controls, .compass, .stage-legend")) return;
    host.setPointerCapture(e.pointerId);
    drag = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty };
    host.classList.add("dragging");
  });
  host.addEventListener("pointermove", (e) => {
    if (!drag) return;
    view.tx = drag.tx + (e.clientX - drag.x);
    view.ty = drag.ty + (e.clientY - drag.y);
    applyView();
  });
  const endDrag = (e) => {
    if (!drag) return;
    drag = null;
    host.classList.remove("dragging");
    try { host.releasePointerCapture(e.pointerId); } catch (_) {}
  };
  host.addEventListener("pointerup", endDrag);
  host.addEventListener("pointercancel", endDrag);
  host.addEventListener("dblclick", (e) => {
    if (e.target.closest("#zoom-controls")) return;
    e.preventDefault();
    resetView();
  });

  // 双指缩放
  let pinch = null;
  host.addEventListener("touchstart", (e) => {
    if (e.touches.length === 2) {
      e.preventDefault();
      const [a, b] = e.touches;
      pinch = {
        dist: Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY),
        scale: view.scale,
        mx: (a.clientX + b.clientX) / 2,
        my: (a.clientY + b.clientY) / 2,
        tx: view.tx, ty: view.ty,
      };
      drag = null;
    }
  }, { passive: false });
  host.addEventListener("touchmove", (e) => {
    if (!pinch || e.touches.length !== 2) return;
    e.preventDefault();
    const [a, b] = e.touches;
    const dist = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
    const factor = dist / Math.max(pinch.dist, 1e-6);
    let next = Math.min(view.maxScale, Math.max(view.minScale, pinch.scale * factor));
    const rect = host.getBoundingClientRect();
    const px = pinch.mx - rect.left;
    const py = pinch.my - rect.top;
    const cx = (px - pinch.tx) / pinch.scale;
    const cy = (py - pinch.ty) / pinch.scale;
    const mx = (a.clientX + b.clientX) / 2 - rect.left;
    const my = (a.clientY + b.clientY) / 2 - rect.top;
    view.scale = next;
    view.tx = mx - cx * next;
    view.ty = my - cy * next;
    applyView();
  }, { passive: false });
  host.addEventListener("touchend", () => { pinch = null; });
  host.addEventListener("touchcancel", () => { pinch = null; });

  $("zoom-in")?.addEventListener("click", () => zoomBy(1.25));
  $("zoom-out")?.addEventListener("click", () => zoomBy(1 / 1.25));
  $("zoom-reset")?.addEventListener("click", () => resetView());
  host.addEventListener("keydown", (e) => {
    if (e.key === "+" || e.key === "=") { e.preventDefault(); zoomBy(1.2); }
    else if (e.key === "-" || e.key === "_") { e.preventDefault(); zoomBy(1 / 1.2); }
    else if (e.key === "0") { e.preventDefault(); resetView(); }
  });

  window.addEventListener("resize", () => {
    if (!state.dem) return;
    const hostEl = $("stage-canvas");
    if (!hostEl || !view.baseW) {
      fitLayerStack(true);
      return;
    }
    const fx = (hostEl.clientWidth / 2 - view.tx) / (view.baseW * view.scale || 1);
    const fy = (hostEl.clientHeight / 2 - view.ty) / (view.baseH * view.scale || 1);
    const prevScale = view.scale;
    const ar = (state.dem.width || 1) / (state.dem.height || 1);
    const availW = Math.max(40, hostEl.clientWidth - 16);
    const availH = Math.max(40, hostEl.clientHeight - 16);
    let w = availW, h = w / ar;
    if (h > availH) { h = availH; w = h * ar; }
    view.baseW = Math.floor(w);
    view.baseH = Math.floor(h);
    view.scale = prevScale;
    view.tx = hostEl.clientWidth / 2 - fx * view.baseW * view.scale;
    view.ty = hostEl.clientHeight / 2 - fy * view.baseH * view.scale;
    applyView();
  });
}
