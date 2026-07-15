/** DOM / 格式化工具 */
export const $ = (id) => document.getElementById(id);

export function setStatus(msg, kind) {
  const el = $("status-text");
  if (!el) return;
  el.textContent = msg || "";
  el.className = kind || "";
  el.id = "status-text";
  if (kind) el.classList.add(kind);
}

export function fmt(v, n) {
  if (v === null || v === undefined || Number.isNaN(+v)) return "—";
  return Number(v).toFixed(n);
}

export function esc(s) {
  return String(s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

export function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

export function base64ToBytes(b64) {
  const bin = atob(b64);
  const arr = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
  return arr;
}
