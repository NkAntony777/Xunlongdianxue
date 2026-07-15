/** 分析结果：图层叠合、SVG 标注、右侧面板 */
import {
  BEAST_NAMES, BEAST_COLORS, BEAST_RING, BEAST_SOFT, BEAST_SHORT, BEAST_ORDER,
  FACING_METHOD_LABEL,
} from "./config.js";
import { state } from "./state.js";
import { $, fmt, esc, setStatus } from "./utils.js";
import { fitLayerStack, showZoomHint } from "./analysis-map.js";
import { fetchFourBeastsAt } from "./api.js";
import {
  storeLayerImagesFromPayload,
  refreshResultMapOverlays,
  updateResultMapAnnotations,
  syncResultMap,
} from "./result-map.js";

export function applyAll(layers, elevMeta) {
  state.bbox = layers.bbox;
  state.dem = layers.dem;
  if (elevMeta && elevMeta.bbox_lonlat) {
    state.bboxLonLat = elevMeta.bbox_lonlat;
  }
  const fb = layers.structured?.four_beasts || null;
  state.fb = fb;
  state.fbPeak = fb ? JSON.parse(JSON.stringify(fb)) : null;
  state.centerXY = layers.structured?.center || null;
  state.centerXYPeak = state.centerXY ? { ...state.centerXY } : null;
  state.facing = layers.structured?.facing || 0;
  state.sit = layers.structured?.sit ?? ((state.facing + 180) % 360);
  state.facingMethod = layers.structured?.facing_method || "";
  state.fbMeta = layers.structured?.four_beasts_meta || null;
  state.candidates = layers.structured?.candidates || [];
  state.selectedCand = null;
  state.beastsSource = "peak";
  state.beastsLoading = false;
  state.fbCache = {
    __peak__: {
      four_beasts: state.fbPeak,
      center: state.centerXYPeak,
      facing: state.facing,
      sit: state.sit,
      facing_method: state.facingMethod,
      meta: state.fbMeta,
    },
  };
  // 场评最高点：优先后端 qi 峰 / score.peak；穴心 center 可为龙 top1
  const center = layers.structured?.center;
  if (center && center.x != null && center.y != null) {
    state.centerXY = center;
    state.centerXYPeak = { ...center };
  }
  state.peakXY = layers.score?.peak_xy || layers.score?.legend?.peak_xy || null;
  // 若 structured 标明场评峰中心，与穴心一致
  if (
    layers.structured?.center_source === "score_field_peak"
    && center && center.x != null
  ) {
    state.peakXY = [center.x, center.y];
  }
  if (!state.peakXY && center && center.x != null) {
    state.peakXY = [center.x, center.y];
  }
  if (!state.peakXY && state.candidates.length) {
    // 带 is_qi_peak 的候选优先
    const qp = state.candidates.find((c) => c.meta?.is_qi_peak);
    const c0 = qp || state.candidates[0];
    state.peakXY = [c0.x, c0.y];
  }
  const ridgeFeats = layers.structured?.ridges_geojson?.features || [];
  state.ridges = ridgeFeats.map((f) => ({ coords: f.geometry.coordinates }));

  // 有候选时默认打开候选图层，便于点选看四象
  if (state.candidates.length) {
    state.layerVisible.candidates = true;
    const btn = document.querySelector('#analysis-toggles [data-layer="candidates"]');
    if (btn) btn.classList.add("active");
  }

  fitLayerStack(true);
  applyLayerImages(layers);
  renderOverlay();
  renderDemInfo(layers, elevMeta);
  renderBeastList();
  renderCandList();
  renderCandDetail();
  wireOverlayClick();
  const arrow = $("compass-arrow");
  if (arrow) arrow.style.transform = `rotate(${state.facing || 0}deg)`;
  showZoomHint();
  // 若用户已在高德对照模式，同步叠加热力与穴点
  if (state.displayMode === "gaode") {
    syncResultMap(true);
  }
}

export function applyLayerImages(d) {
  const set = (id, payload) => {
    const el = $(id);
    if (!el) return;
    if (payload && payload.png_base64) {
      el.src = "data:image/png;base64," + payload.png_base64;
      el.style.display = "block";
    } else {
      el.removeAttribute("src");
      el.style.display = "none";
    }
  };
  set("layer-basemap", d.basemap);
  set("layer-contours", d.contours);
  set("layer-water", d.water);
  set("layer-influence", d.water_influence);
  set("layer-buildings", d.buildings);
  set("layer-score", d.score);
  storeLayerImagesFromPayload(d);
  applyLayerVisibility();
}

export function applyLayerVisibility() {
  const v = state.layerVisible;
  const map = {
    "layer-score": v.score,
    "layer-contours": v.contours,
    "layer-water": v.water,
    "layer-influence": v.influence,
    "layer-buildings": v.buildings,
  };
  for (const [id, on] of Object.entries(map)) {
    const el = $(id);
    if (!el) continue;
    if (!el.getAttribute("src")) { el.style.display = "none"; continue; }
    el.style.display = on ? "block" : "none";
  }
  if (state.displayMode === "gaode") {
    refreshResultMapOverlays();
  }
}

function worldToImg(x, y) {
  const [minx, miny, maxx, maxy] = state.bbox;
  const W = state.dem.width, H = state.dem.height;
  return [
    (x - minx) / (maxx - minx) * W,
    (maxy - y) / (maxy - miny) * H,
  ];
}

function imgToWorld(px, py) {
  const [minx, miny, maxx, maxy] = state.bbox;
  const W = state.dem.width, H = state.dem.height;
  return [
    minx + (px / W) * (maxx - minx),
    maxy - (py / H) * (maxy - miny),
  ];
}

function applyFbPayload(data, source) {
  state.fb = data.four_beasts || null;
  state.centerXY = data.center || null;
  state.facing = data.facing ?? 0;
  state.sit = data.sit ?? ((state.facing + 180) % 360);
  state.facingMethod = data.facing_method || "";
  state.fbMeta = data.meta || null;
  state.beastsSource = source;
  const arrow = $("compass-arrow");
  if (arrow) arrow.style.transform = `rotate(${state.facing || 0}deg)`;
}

function restorePeakBeasts() {
  state.selectedCand = null;
  state.beastsLoading = false;
  if (state.fbCache.__peak__) {
    applyFbPayload(state.fbCache.__peak__, "peak");
  } else if (state.fbPeak) {
    state.fb = JSON.parse(JSON.stringify(state.fbPeak));
    state.centerXY = state.centerXYPeak ? { ...state.centerXYPeak } : null;
    state.beastsSource = "peak";
  }
  renderBeastList();
  renderCandList();
  renderCandDetail();
  renderOverlay();
  updateResultMapAnnotations();
}

/** 选中候选穴：拉四象并刷新图/侧栏；再点取消则回到场评最高点 */
export async function selectCandidate(candId) {
  if (!candId) {
    restorePeakBeasts();
    return;
  }
  if (state.selectedCand === candId) {
    restorePeakBeasts();
    return;
  }
  const c = state.candidates.find((x) => x.id === candId);
  if (!c) return;

  // 缓存场评峰值四象（首次）
  if (!state.fbCache.__peak__ && state.fbPeak) {
    state.fbCache.__peak__ = {
      four_beasts: state.fbPeak,
      center: state.centerXYPeak,
      facing: state.facing,
      sit: state.sit,
      facing_method: state.facingMethod,
      meta: state.fbMeta,
    };
  }

  state.selectedCand = candId;
  state.layerVisible.candidates = true;
  const btn = document.querySelector('#analysis-toggles [data-layer="candidates"]');
  if (btn) btn.classList.add("active");

  if (state.fbCache[candId]) {
    applyFbPayload(state.fbCache[candId], "candidate");
    state.beastsLoading = false;
    renderBeastList();
    renderCandList();
    renderCandDetail();
    renderOverlay();
    updateResultMapAnnotations();
    setStatus(`四象：${c.id}（缓存）`, "ok");
    return;
  }

  state.beastsLoading = true;
  renderBeastList();
  renderCandList();
  renderOverlay();
  updateResultMapAnnotations();
  setStatus(`计算 ${c.id} 的四象…`);

  try {
    const data = await fetchFourBeastsAt(
      state.demPath, state.waterPath, c.x, c.y,
    );
    state.fbCache[candId] = data;
    // 若用户在等待期间已点了别的，忽略
    if (state.selectedCand !== candId) return;
    applyFbPayload(data, "candidate");
    state.beastsLoading = false;
    renderBeastList();
    renderCandList();
    renderCandDetail();
    renderOverlay();
    updateResultMapAnnotations();
    setStatus(`四象已更新 · 穴心 ${c.id}`, "ok");
  } catch (e) {
    console.error(e);
    state.beastsLoading = false;
    state.selectedCand = null;
    restorePeakBeasts();
    setStatus("四象失败: " + (e.message || e), "err");
  }
}

function beastMarkerSvg(k, px, py, s) {
  // 仅圆形标识（中心一字）；右侧面板已有全名，不再叠矩形标签
  const fill = BEAST_COLORS[k];
  const ring = BEAST_RING[k] || "#fff";
  const short = BEAST_SHORT[k] || "·";
  const isMajor = k === "xuanwu" || k === "shaozu";
  const r = isMajor ? s * 1.45 : s * 1.2;
  const fs = s * 0.95;

  let html = "";
  html += `<circle cx="${px.toFixed(2)}" cy="${py.toFixed(2)}" r="${(r * 1.55).toFixed(2)}" fill="${fill}" opacity="0.1"/>`;
  html += `<circle cx="${px.toFixed(2)}" cy="${py.toFixed(2)}" r="${r.toFixed(2)}" fill="#fff" stroke="${ring}" stroke-width="${Math.max(1.6, s * 0.32)}" opacity="0.98"/>`;
  html += `<circle cx="${px.toFixed(2)}" cy="${py.toFixed(2)}" r="${(r * 0.72).toFixed(2)}" fill="${fill}" stroke="#fff" stroke-width="${Math.max(0.9, s * 0.14)}"/>`;
  html += `<text x="${px.toFixed(2)}" y="${(py + fs * 0.38).toFixed(2)}" font-size="${fs.toFixed(2)}" font-weight="800" fill="#fff" text-anchor="middle" font-family="system-ui,sans-serif" style="pointer-events:none">${short}</text>`;
  return html;
}

function axisLine(x1, y1, x2, y2, color, sw, dash, op) {
  const d = dash ? ` stroke-dasharray="${dash}"` : "";
  return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="${sw}" opacity="${op}" stroke-linecap="round"${d}/>`;
}

export function renderOverlay() {
  const svg = $("overlay");
  if (!svg || !state.dem || !state.bbox) {
    if (svg) svg.innerHTML = "";
    return;
  }
  const W = state.dem.width, H = state.dem.height;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  const s = Math.sqrt(W * H) / 55;
  const sw = Math.max(1.0, s * 0.22);
  let html = "";
  // SVG defs：阴影
  html += `<defs>
    <filter id="beastShadow" x="-30%" y="-30%" width="160%" height="160%">
      <feDropShadow dx="0" dy="0.6" stdDeviation="0.9" flood-color="#0f172a" flood-opacity="0.18"/>
    </filter>
  </defs>`;
  const v = state.layerVisible;

  if (v.ridges && state.ridges.length) {
    state.ridges.forEach((r, idx) => {
      if (!r.coords || r.coords.length < 2) return;
      const pts = r.coords
        .map(([x, y]) => worldToImg(x, y).map((vv) => vv.toFixed(1)).join(","))
        .join(" ");
      html += `<polyline points="${pts}" fill="none" stroke="#6366f1" stroke-width="${Math.max(1, s * 0.35 - idx * 0.2)}" stroke-opacity="0.4" stroke-dasharray="${idx === 0 ? "0" : "5,4"}"/>`;
    });
  }

  if (v.beasts && state.fb) {
    const hub = state.centerXY
      || (state.peakXY ? { x: state.peakXY[0], y: state.peakXY[1] } : null)
      || state.fb.xuanwu;
    if (hub) {
      const [ex, ey] = worldToImg(hub.x, hub.y);
      // 主轴光晕 + 线
      if (state.fb.shaozu) {
        const [sx, sy] = worldToImg(state.fb.shaozu.x, state.fb.shaozu.y);
        html += axisLine(sx, sy, ex, ey, "#94a3b8", Math.max(2.4, s * 0.58), null, 0.22);
        html += axisLine(sx, sy, ex, ey, "#334155", Math.max(1.3, s * 0.34), "8,5", 0.88);
      }
      if (state.fb.xuanwu) {
        const [ux, uy] = worldToImg(state.fb.xuanwu.x, state.fb.xuanwu.y);
        html += axisLine(ux, uy, ex, ey, "#0f172a", Math.max(1.4, s * 0.36), null, 0.82);
      }
      if (state.fb.zhuque) {
        const [zx, zy] = worldToImg(state.fb.zhuque.x, state.fb.zhuque.y);
        html += axisLine(ex, ey, zx, zy, "#dc2626", Math.max(1.5, s * 0.38), null, 0.78);
      }
      for (const k of ["qinglong", "baihu"]) {
        const b = state.fb[k];
        if (!b) continue;
        const [bx, by] = worldToImg(b.x, b.y);
        html += axisLine(ex, ey, bx, by, BEAST_COLORS[k], Math.max(1.1, s * 0.28), "5,4", 0.72);
      }
      // 穴心十字
      const cr = s * 0.85;
      html += `<circle cx="${ex.toFixed(2)}" cy="${ey.toFixed(2)}" r="${(cr * 1.15).toFixed(2)}" fill="none" stroke="#f59e0b" stroke-width="${Math.max(1.2, s * 0.22)}" opacity="0.85"/>`;
      html += `<circle cx="${ex.toFixed(2)}" cy="${ey.toFixed(2)}" r="${(cr * 0.45).toFixed(2)}" fill="#f59e0b" opacity="0.9"/>`;
    }
    for (const k of BEAST_ORDER) {
      const b = state.fb[k];
      if (!b || b.x == null || b.y == null) continue;
      if (!Number.isFinite(b.x) || !Number.isFinite(b.y)) continue;
      const [px, py] = worldToImg(b.x, b.y);
      if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
      if (px < -2 || py < -2 || px > W + 2 || py > H + 2) continue;
      html += beastMarkerSvg(k, px, py, s);
    }
  }

  // 场评最高点（未选候选时强调；选了候选则淡化）
  if (v.peak && state.peakXY) {
    const [px, py] = worldToImg(state.peakXY[0], state.peakXY[1]);
    const peakOp = state.selectedCand ? 0.45 : 0.95;
    html += `<circle cx="${px.toFixed(2)}" cy="${py.toFixed(2)}" r="${(s * 1.8).toFixed(2)}" fill="none" stroke="#e07a32" stroke-width="${Math.max(1.4, s * 0.28)}" opacity="${peakOp}"/>`;
    html += `<circle cx="${px.toFixed(2)}" cy="${py.toFixed(2)}" r="${(s * 1.15).toFixed(2)}" fill="#f0a040" stroke="#fff" stroke-width="${sw.toFixed(2)}" opacity="${peakOp}"/>`;
    if (!state.selectedCand) {
      html += `<text x="${px.toFixed(2)}" y="${(py - s * 2.4).toFixed(2)}" font-size="${(s * 1.15).toFixed(2)}" font-weight="700" fill="#c45a10" text-anchor="middle" stroke="#fff" stroke-width="${(sw * 1.3).toFixed(2)}" paint-order="stroke">场评最高点</text>`;
    }
  }

  if (v.candidates && state.candidates.length) {
    state.candidates.forEach((c, idx) => {
      if (c.x == null || c.y == null || !Number.isFinite(c.x) || !Number.isFinite(c.y)) {
        return; // 无效坐标勿画到左上角
      }
      const [px, py] = worldToImg(c.x, c.y);
      if (!Number.isFinite(px) || !Number.isFinite(py)) return;
      if (px < -2 || py < -2 || px > W + 2 || py > H + 2) return;
      const isSel = state.selectedCand === c.id;
      // 明堂橙心必须显示候选（不再因靠近场评最高点而隐藏）
      const r = isSel ? s * 1.25 : s * 0.78;
      const rank = c.rank || (idx + 1);
      html += `<circle class="cand-hit" data-cand="${esc(c.id)}" cx="${px.toFixed(2)}" cy="${py.toFixed(2)}" r="${(r * 2.2).toFixed(2)}" fill="transparent" style="cursor:pointer"/>`;
      if (isSel) {
        html += `<circle cx="${px.toFixed(2)}" cy="${py.toFixed(2)}" r="${(r * 1.55).toFixed(2)}" fill="#dc2626" opacity="0.15"/>`;
        html += `<circle cx="${px.toFixed(2)}" cy="${py.toFixed(2)}" r="${r.toFixed(2)}" fill="#dc2626" stroke="#fff" stroke-width="${Math.max(1.6, sw).toFixed(2)}" opacity="0.95"/>`;
        html += `<text x="${px.toFixed(2)}" y="${(py + s * 0.38).toFixed(2)}" font-size="${(s * 0.9).toFixed(2)}" font-weight="800" fill="#fff" text-anchor="middle" style="pointer-events:none">${rank}</text>`;
        html += `<text x="${px.toFixed(2)}" y="${(py - s * 1.9).toFixed(2)}" font-size="${(s * 1.05).toFixed(2)}" font-weight="700" fill="#b91c1c" text-anchor="middle" stroke="#fff" stroke-width="${sw.toFixed(2)}" paint-order="stroke" style="pointer-events:none">${esc(c.id)}</text>`;
      } else {
        html += `<circle cx="${px.toFixed(2)}" cy="${py.toFixed(2)}" r="${r.toFixed(2)}" fill="#fff" stroke="#e11d48" stroke-width="${Math.max(1.3, sw).toFixed(2)}" opacity="0.92"/>`;
        html += `<text x="${px.toFixed(2)}" y="${(py + s * 0.32).toFixed(2)}" font-size="${(s * 0.72).toFixed(2)}" font-weight="700" fill="#e11d48" text-anchor="middle" style="pointer-events:none">${rank}</text>`;
      }
    });
  }

  // 加载中提示
  if (state.beastsLoading) {
    html += `<rect x="${(W * 0.28).toFixed(1)}" y="${(H * 0.04).toFixed(1)}" width="${(W * 0.44).toFixed(1)}" height="${(s * 2.8).toFixed(1)}" rx="${(s * 0.5).toFixed(1)}" fill="rgba(15,23,42,0.78)"/>`;
    html += `<text x="${(W * 0.5).toFixed(1)}" y="${(H * 0.04 + s * 1.85).toFixed(1)}" font-size="${(s * 1.15).toFixed(2)}" fill="#f8fafc" text-anchor="middle" font-weight="600">正在计算选中候选的四象…</text>`;
  }

  svg.innerHTML = html;
}

let _overlayWired = false;
function wireOverlayClick() {
  const svg = $("overlay");
  if (!svg || _overlayWired) return;
  _overlayWired = true;
  // 仅 .cand-hit 可点（CSS pointer-events），不挡画布平移
  svg.addEventListener("click", (ev) => {
    const hit = ev.target.closest?.(".cand-hit");
    if (hit && hit.dataset.cand) {
      ev.stopPropagation();
      selectCandidate(hit.dataset.cand);
    }
  });
}

export function renderDemInfo(layers, elevMeta) {
  const d = layers.dem || {};
  const bbox = layers.bbox || [];
  let bboxLine = "";
  if (elevMeta && elevMeta.bbox_lonlat) {
    const b = elevMeta.bbox_lonlat;
    bboxLine = `${fmt(b[0], 6)}, ${fmt(b[1], 6)}, ${fmt(b[2], 6)}, ${fmt(b[3], 6)}`;
  } else if (d.crs && String(d.crs).includes("4326")) {
    bboxLine = `${fmt(bbox[0], 6)}, ${fmt(bbox[1], 6)}, ${fmt(bbox[2], 6)}, ${fmt(bbox[3], 6)}`;
  } else if (state.mode === "demo") {
    bboxLine = `105.850000, 31.500000, 106.000000, 31.650000`;
  } else {
    bboxLine = bbox.map((v) => fmt(v, 1)).join(", ");
  }
  const sizeLine = (d.width_m && d.height_m)
    ? `${Math.round(d.width_m)}m × ${Math.round(d.height_m)}m` : "—";
  if ($("dem-info")) {
    $("dem-info").innerHTML = `
      <div><span class="k">bbox:</span> ${bboxLine}</div>
      <div><span class="k">网格:</span> ${d.width || "—"} × ${d.height || "—"}</div>
      <div><span class="k">真实宽高:</span> ${sizeLine}</div>
      <div><span class="k">高程:</span> ${fmt(d.vmin, 0)}m – ${fmt(d.vmax, 0)}m</div>
    `;
  }
}

function xyToColRow(x, y) {
  const [minx, miny, maxx, maxy] = state.bbox;
  const col = Math.round((x - minx) / (maxx - minx) * (state.dem.width - 1));
  const row = Math.round((maxy - y) / (maxy - miny) * (state.dem.height - 1));
  return { col, row };
}

export function renderBeastList() {
  const el = $("beast-list");
  if (!el) return;
  if (!state.fb || !state.dem || !state.bbox) {
    el.innerHTML = `<div class="empty">尚未分析</div>`;
    return;
  }
  const method = FACING_METHOD_LABEL[state.facingMethod] || state.facingMethod || "—";
  let sourceLabel = "场评最高点";
  if (state.beastsSource === "candidate" && state.selectedCand) {
    sourceLabel = `候选 ${state.selectedCand}`;
  } else if (state.beastsLoading) {
    sourceLabel = "计算中…";
  }

  let html = `
    <div class="beast-legend">
      ${BEAST_ORDER.map((k) => `
        <span class="beast-legend-item ${k}">
          <i class="beast-legend-dot"></i>${BEAST_NAMES[k]}
        </span>`).join("")}
    </div>
    <div class="beast-meta">
      <div class="beast-source-row">
        <span class="beast-source-tag ${state.beastsSource}">${esc(sourceLabel)}</span>
        ${state.selectedCand ? `<button type="button" class="beast-reset-btn" id="btn-reset-peak-beasts">回到最高点</button>` : ""}
      </div>
      <div><span class="k">朝向</span> ${fmt(state.facing, 0)}°（北=0）</div>
      <div><span class="k">坐向</span> ${fmt(state.sit, 0)}°</div>
      <div><span class="k">推断</span> ${esc(method)}</div>
    </div>`;

  if (state.beastsLoading) {
    html += `<div class="beast-loading">正在按选中候选重算四象…</div>`;
  }

  for (const k of BEAST_ORDER) {
    const b = state.fb[k];
    if (!b) {
      html += `<div class="beast-card missing">
        <span class="beast-badge ${k}"><i class="beast-pip"></i>${BEAST_NAMES[k]}</span>
        <span class="beast-miss">未识别</span>
      </div>`;
      continue;
    }
    const { col, row } = xyToColRow(b.x, b.y);
    const dist = b.dist_m != null ? Math.round(b.dist_m) + " m" : "—";
    const elev = b.elev_m != null ? Math.round(b.elev_m) + " m" : "—";
    const onRidge = b.on_ridge ? `<span class="beast-chip ridge">脊上</span>` : "";
    html += `<div class="beast-card ${k}">
      <div class="beast-card-head">
        <span class="beast-badge ${k}"><i class="beast-pip"></i>${BEAST_NAMES[k]}</span>
        <span class="beast-metrics">${dist}<span class="sep">·</span>${elev}</span>
      </div>
      <div class="beast-card-sub">栅格 (${col}, ${row}) ${onRidge}</div>
    </div>`;
  }
  el.innerHTML = html;

  $("btn-reset-peak-beasts")?.addEventListener("click", () => {
    restorePeakBeasts();
    setStatus("四象已回到场评最高点", "ok");
  });
}

export function renderCandList() {
  const el = $("cand-list");
  if (!el) return;
  if (!state.candidates.length) {
    el.innerHTML = `<div class="empty">暂无候选穴</div>`;
    if ($("cand-detail")) $("cand-detail").style.display = "none";
    return;
  }
  el.innerHTML = `
    <div class="cand-hint">点击候选可切换四象（再点取消）</div>
    ${state.candidates.slice(0, 10).map((c) => `
    <div class="cand-row${state.selectedCand === c.id ? " active" : ""}" data-id="${esc(c.id)}">
      <span><span class="rank">${c.rank}</span>${esc(c.id)} · ${esc(c.form_type || "—")}</span>
      <span><b>${Math.round(c.overall_score)}</b></span>
    </div>`).join("")}`;
  el.querySelectorAll(".cand-row").forEach((n) => {
    n.addEventListener("click", () => {
      selectCandidate(n.dataset.id);
    });
  });
}

export function renderCandDetail() {
  const detail = $("cand-detail");
  if (!detail) return;
  const c = state.candidates.find((x) => x.id === state.selectedCand);
  if (!c) { detail.style.display = "none"; return; }
  const scores = c.scores || {};
  const geo = c.geography || {};
  const lines = [];
  lines.push(`<div><span class="form-type">${esc(c.form_type || "—")}</span> <span style="color:#888">${esc(c.id)} · 综合 ${Math.round(c.overall_score)}</span></div>`);
  lines.push(`<div class="cand-detail-tip">当前四象以该候选为穴心重算</div>`);
  lines.push(`<div style="margin-top:4px;color:#555">评分</div>`);
  for (const [k, v] of Object.entries(scores)) {
    lines.push(`<div class="kv"><span class="k">${esc(k)}</span><span>${Number(v).toFixed(0)}</span></div>`);
  }
  if (Object.keys(geo).length) {
    lines.push(`<div style="margin-top:4px;color:#555">地理</div>`);
    for (const [k, v] of Object.entries(geo)) {
      if (v === null || v === undefined) continue;
      const s = typeof v === "number" ? Number(v).toFixed(1) : v;
      lines.push(`<div class="kv"><span class="k">${esc(k)}</span><span>${esc(String(s))}</span></div>`);
    }
  }
  detail.innerHTML = lines.join("");
  detail.style.display = "block";
}

export function wireLayerToggles() {
  $("basemap-modes")?.addEventListener("click", async (ev) => {
    const btn = ev.target.closest("button.pill");
    if (!btn) return;
    [...$("basemap-modes").children].forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.basemapMode = btn.dataset.mode;
    if (typeof wireLayerToggles.onBasemapChange === "function") {
      wireLayerToggles.onBasemapChange();
    }
  });

  function wire(containerId) {
    $(containerId)?.addEventListener("click", (ev) => {
      const btn = ev.target.closest("button.pill");
      if (!btn) return;
      btn.classList.toggle("active");
      const layer = btn.dataset.layer;
      state.layerVisible[layer] = btn.classList.contains("active");
      if (["beasts", "candidates", "ridges", "peak"].includes(layer)) {
        renderOverlay();
        updateResultMapAnnotations();
      } else {
        applyLayerVisibility();
      }
    });
  }
  wire("analysis-primary");
  wire("analysis-toggles");
}
