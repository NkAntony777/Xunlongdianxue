/**
 * 分析结果 — 高德对照底图
 * 将评分热力等 PNG 叠在高德瓦片上，并标出候选穴经纬度。
 */
import { state } from "./state.js";
import { $ } from "./utils.js";
import {
  isGeographicCrs,
  isWebMercator,
  mercatorToLonLat,
  lonLatToMercator,
  resolveLonLatBounds as resolveLonLatBoundsPure,
  worldToLonLat as worldToLonLatPure,
} from "./geo.js";

// 再导出纯函数，便于调试与兼容旧引用
export {
  isGeographicCrs, isWebMercator, mercatorToLonLat, lonLatToMercator,
};

let map = null;
let overlays = {};
let candLayer = null;
let peakMarker = null;
let aoiCircle = null;
let beastLayer = null;
let ridgeLayer = null;
let wired = false;
let lastBoundsKey = "";
/** 由 app/render-ui 注入，避免循环依赖 */
let onCandidateSelect = null;

export function setCandidateSelectHandler(fn) {
  onCandidateSelect = fn;
}

function geoCtx() {
  return {
    bboxLonLat: state.bboxLonLat,
    bbox: state.bbox,
    crs: state.dem?.crs,
    mode: state.mode,
  };
}

/** 解析有效经纬度包围盒 [minLon, minLat, maxLon, maxLat] */
export function resolveLonLatBounds() {
  return resolveLonLatBoundsPure(geoCtx());
}

/** 分析世界坐标 → WGS84 */
export function worldToLonLat(x, y) {
  return worldToLonLatPure(x, y, geoCtx());
}

function pngToUrl(payload) {
  if (!payload) return null;
  if (typeof payload === "string") {
    if (payload.startsWith("data:")) return payload;
    return "data:image/png;base64," + payload;
  }
  if (payload.png_base64) return "data:image/png;base64," + payload.png_base64;
  return null;
}

function leafletBounds(llb) {
  // Leaflet: [[south, west], [north, east]]
  return L.latLngBounds(
    [llb[1], llb[0]],
    [llb[3], llb[2]],
  );
}

function ensureMap() {
  if (map) return map;
  const el = $("result-amap");
  if (!el || typeof L === "undefined") return null;
  map = L.map(el, {
    center: [state.center.lat, state.center.lon],
    zoom: 12,
    zoomControl: true,
    attributionControl: true,
  });
  L.tileLayer(
    "https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}",
    {
      maxZoom: 18,
      subdomains: "1234",
      attribution: "© 高德",
    },
  ).addTo(map);
  candLayer = L.layerGroup().addTo(map);
  beastLayer = L.layerGroup().addTo(map);
  ridgeLayer = L.layerGroup().addTo(map);
  return map;
}

function setImageOverlay(key, url, opacity, zIndex) {
  if (!map) return;
  if (overlays[key]) {
    map.removeLayer(overlays[key]);
    overlays[key] = null;
  }
  const llb = resolveLonLatBounds();
  if (!url || !llb) return;
  const layer = L.imageOverlay(url, leafletBounds(llb), {
    opacity: opacity ?? 0.75,
    interactive: false,
    zIndex: zIndex ?? 200,
  });
  overlays[key] = layer;
  layer.addTo(map);
}

function clearOverlays() {
  if (!map) return;
  for (const k of Object.keys(overlays)) {
    if (overlays[k]) {
      map.removeLayer(overlays[k]);
      overlays[k] = null;
    }
  }
}

function candIcon(rank, selected) {
  return L.divIcon({
    className: "result-cand-marker",
    html: `<div class="result-cand-pin${selected ? " selected" : ""}">${rank}</div>`,
    iconSize: selected ? [34, 34] : [28, 28],
    iconAnchor: selected ? [17, 17] : [14, 14],
  });
}

function rebuildCandidates() {
  if (!candLayer) return;
  candLayer.clearLayers();
  if (!state.layerVisible.candidates || !state.candidates.length) return;

  state.candidates.forEach((c, idx) => {
    const ll = worldToLonLat(c.x, c.y);
    if (!ll) return;
    const rank = c.rank || (idx + 1);
    const selected = state.selectedCand === c.id;
    const m = L.marker([ll.lat, ll.lon], {
      icon: candIcon(rank, selected),
      zIndexOffset: selected ? 800 : 400 + (10 - Math.min(rank, 10)),
    });
    const score = c.overall_score != null ? Math.round(c.overall_score) : "—";
    const form = c.form_type || "—";
    m.bindPopup(
      `<div style="min-width:140px">
        <b>候选穴 ${escHtml(c.id)}</b><br/>
        排名 #${rank} · 综合 <b>${score}</b><br/>
        形态：${escHtml(form)}<br/>
        <span style="color:#666;font-size:11px">${ll.lat.toFixed(5)}, ${ll.lon.toFixed(5)}</span>
      </div>`,
    );
    m.on("click", () => {
      if (typeof onCandidateSelect === "function") onCandidateSelect(c.id);
    });
    m.addTo(candLayer);
  });
}

function rebuildPeak() {
  if (peakMarker && map) {
    map.removeLayer(peakMarker);
    peakMarker = null;
  }
  if (!map || !state.layerVisible.peak || !state.peakXY) return;
  const ll = worldToLonLat(state.peakXY[0], state.peakXY[1]);
  if (!ll) return;
  peakMarker = L.marker([ll.lat, ll.lon], {
    icon: L.divIcon({
      className: "result-cand-marker",
      html: '<div class="result-peak-pin" title="场评最高点"></div>',
      iconSize: [14, 14],
      iconAnchor: [7, 7],
    }),
    zIndexOffset: 300,
    interactive: true,
  });
  peakMarker.bindTooltip("场评最高点", { direction: "top", offset: [0, -6] });
  peakMarker.addTo(map);
}

function rebuildBeasts() {
  if (!beastLayer) return;
  beastLayer.clearLayers();
  if (!state.layerVisible.beasts || !state.fb) return;

  const colors = {
    shaozu: "#475569", xuanwu: "#0f172a", zhuque: "#dc2626",
    qinglong: "#0d9488", baihu: "#d97706",
  };
  const names = {
    shaozu: "少祖", xuanwu: "玄武", zhuque: "朱雀",
    qinglong: "青龙", baihu: "白虎",
  };
  const hub = state.centerXY
    || (state.peakXY ? { x: state.peakXY[0], y: state.peakXY[1] } : null)
    || state.fb.xuanwu;
  const hubLl = hub ? worldToLonLat(hub.x, hub.y) : null;

  for (const [k, b] of Object.entries(state.fb)) {
    if (!b || b.x == null || b.y == null) continue;
    const ll = worldToLonLat(b.x, b.y);
    if (!ll) continue;
    const col = colors[k] || "#666";
    L.circleMarker([ll.lat, ll.lon], {
      radius: 7, color: "#fff", weight: 2, fillColor: col, fillOpacity: 0.95,
    })
      .bindTooltip(names[k] || k, { direction: "top" })
      .addTo(beastLayer);
    if (hubLl) {
      L.polyline(
        [[hubLl.lat, hubLl.lon], [ll.lat, ll.lon]],
        { color: col, weight: 1.5, opacity: 0.7, dashArray: k === "shaozu" ? "6,4" : null },
      ).addTo(beastLayer);
    }
  }
  if (hubLl) {
    L.circleMarker([hubLl.lat, hubLl.lon], {
      radius: 5, color: "#f59e0b", weight: 2, fillColor: "#f59e0b", fillOpacity: 0.9,
    }).bindTooltip("穴心", { direction: "top" }).addTo(beastLayer);
  }
}

function rebuildRidges() {
  if (!ridgeLayer) return;
  ridgeLayer.clearLayers();
  if (!state.layerVisible.ridges || !state.ridges.length) return;
  state.ridges.forEach((r, idx) => {
    if (!r.coords || r.coords.length < 2) return;
    const latlngs = [];
    for (const [x, y] of r.coords) {
      const ll = worldToLonLat(x, y);
      if (ll) latlngs.push([ll.lat, ll.lon]);
    }
    if (latlngs.length < 2) return;
    L.polyline(latlngs, {
      color: "#6366f1",
      weight: Math.max(2, 4 - idx),
      opacity: 0.55,
      dashArray: idx === 0 ? null : "6,4",
    }).addTo(ridgeLayer);
  });
}

function rebuildAoi() {
  if (aoiCircle && map) {
    map.removeLayer(aoiCircle);
    aoiCircle = null;
  }
  if (!map || !state.center || !state.radius_km) return;
  aoiCircle = L.circle([state.center.lat, state.center.lon], {
    radius: state.radius_km * 1000,
    color: "#1a7f5a",
    weight: 1.5,
    opacity: 0.55,
    fillColor: "#1a7f5a",
    fillOpacity: 0.04,
    dashArray: "4,4",
    interactive: false,
  }).addTo(map);
}

function escHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/** 根据 layerVisible + layerImages 刷新高德叠加层 */
export function refreshResultMapOverlays() {
  if (!map || state.displayMode !== "gaode") return;
  const v = state.layerVisible;
  const img = state.layerImages || {};

  // 不在高德上再叠 DEM 底图（已有路网地名）；只叠分析场
  setImageOverlay("score", v.score ? img.score : null, 0.72, 250);
  setImageOverlay("influence", v.influence ? img.influence : null, 0.55, 230);
  setImageOverlay("buildings", v.buildings ? img.buildings : null, 0.5, 240);
  setImageOverlay("water", v.water ? img.water : null, 0.85, 260);
  setImageOverlay("contours", v.contours ? img.contours : null, 0.7, 270);

  rebuildCandidates();
  rebuildPeak();
  rebuildBeasts();
  rebuildRidges();
}

/** 分析完成后或切到高德时：定位并画全套 */
export function syncResultMap(fitBounds = false) {
  if (state.displayMode !== "gaode") return;
  const m = ensureMap();
  if (!m) return;

  const llb = resolveLonLatBounds();
  if (llb) {
    const key = llb.map((v) => v.toFixed(5)).join(",");
    if (fitBounds || key !== lastBoundsKey) {
      lastBoundsKey = key;
      m.fitBounds(leafletBounds(llb), { padding: [24, 24], maxZoom: 15 });
    }
  } else if (state.center) {
    m.setView([state.center.lat, state.center.lon], 12);
  }

  rebuildAoi();
  refreshResultMapOverlays();
  setTimeout(() => m.invalidateSize(), 80);
}

export function setDisplayMode(mode) {
  const next = mode === "gaode" ? "gaode" : "analysis";
  state.displayMode = next;
  const root = $("analysis-view");
  if (root) {
    root.classList.toggle("mode-gaode", next === "gaode");
    root.classList.toggle("mode-analysis", next !== "gaode");
  }
  document.querySelectorAll("#display-modes .pill").forEach((b) => {
    b.classList.toggle("active", b.dataset.display === next);
  });
  if (next === "gaode") {
    syncResultMap(true);
  }
}

/** 仅刷新标注（选中候选 / 四象切换时） */
export function updateResultMapAnnotations() {
  if (state.displayMode !== "gaode" || !map) return;
  rebuildCandidates();
  rebuildPeak();
  rebuildBeasts();
  rebuildRidges();
}

export function onResultMapShown() {
  if (state.displayMode === "gaode") {
    syncResultMap(false);
  }
}

export function storeLayerImagesFromPayload(d) {
  const pick = (payload) => pngToUrl(payload);
  state.layerImages = {
    basemap: pick(d.basemap),
    score: pick(d.score),
    contours: pick(d.contours),
    water: pick(d.water),
    influence: pick(d.water_influence),
    buildings: pick(d.buildings),
  };
}

export function wireResultMap() {
  if (wired) return;
  wired = true;
  $("display-modes")?.addEventListener("click", (ev) => {
    const btn = ev.target.closest("button.pill[data-display]");
    if (!btn) return;
    setDisplayMode(btn.dataset.display);
  });
  window.addEventListener("resize", () => {
    if (map && state.displayMode === "gaode" && state.view === "analysis") {
      map.invalidateSize();
    }
  });
}
