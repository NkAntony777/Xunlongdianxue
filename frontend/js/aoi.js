/** 高德底图 + 分析区 (AOI) 圈选 */
import { state } from "./state.js";
import { $, setStatus } from "./utils.js";
import { fetchAoiLimits } from "./api.js";

let map = null;
let centerMarker = null;
let bboxCircle = null;
let handleMarker = null;
let onViewChange = null;

export function setViewChangeHandler(fn) {
  onViewChange = fn;
}

export function radiusQuality(r) {
  const L = state.aoiLimits;
  if (r < L.min_radius_km || r > L.max_radius_km) return "invalid";
  if (r < L.recommended_min_km) return "small";
  if (r > L.recommended_max_km) return "large";
  return "ok";
}

export function qualityLabel(q) {
  return { ok: "合适", small: "偏小", large: "偏大", invalid: "无效" }[q] || q;
}

export function qualityColor(q) {
  return { ok: "#1a7f5a", small: "#b7791f", large: "#b7791f", invalid: "#c0392b" }[q] || "#c0392b";
}

export function clampRadius(r) {
  const L = state.aoiLimits;
  return Math.min(L.max_radius_km, Math.max(L.min_radius_km, r));
}

export function updateAoiHud() {
  const q = radiusQuality(state.radius_km);
  const el = $("aoi-quality");
  if (el) {
    el.textContent = qualityLabel(q);
    el.className = q;
  }
  const c = state.center;
  if ($("hud-center")) {
    $("hud-center").textContent = `中心: ${c.lat.toFixed(5)}, ${c.lon.toFixed(5)}`;
  }
  if ($("hud-radius")) {
    $("hud-radius").textContent =
      `半径: ${state.radius_km.toFixed(1)} km（直径 ${(state.radius_km * 2).toFixed(1)} km）`;
  }
  if ($("hud-quality")) {
    const tip = {
      ok: "尺度合适，可覆盖案山/朝山",
      small: `偏小：建议 ≥ ${state.aoiLimits.recommended_min_km} km`,
      large: `偏大：建议 ≤ ${state.aoiLimits.recommended_max_km} km`,
      invalid: `无效：须在 ${state.aoiLimits.min_radius_km}–${state.aoiLimits.max_radius_km} km`,
    }[q];
    $("hud-quality").innerHTML =
      `尺度: <span class="${q === "ok" ? "ok" : q === "invalid" ? "warn" : "soft"}">${qualityLabel(q)}</span> · ${tip}`;
  }
  const btn = $("btn-analyze");
  if (btn) btn.disabled = q === "invalid";
}

export async function loadAoiLimits() {
  try {
    const d = await fetchAoiLimits();
    state.aoiLimits = {
      min_radius_km: d.min_radius_km,
      max_radius_km: d.max_radius_km,
      recommended_min_km: d.recommended_min_km,
      recommended_max_km: d.recommended_max_km,
      default_radius_km: d.default_radius_km,
    };
    const sl = $("radius-slider");
    if (sl) {
      sl.min = d.min_radius_km;
      sl.max = d.max_radius_km;
      sl.step = 0.5;
      if (+sl.value < d.min_radius_km) sl.value = d.default_radius_km;
    }
  } catch (_) {
    /* 使用内置默认 */
  }
}

export function switchView(view) {
  state.view = view;
  $("amap-view")?.classList.toggle("active", view === "map");
  $("analysis-view")?.classList.toggle("active", view === "analysis");
  document.querySelectorAll("#mode-tabs button").forEach((b) => {
    b.classList.toggle("active", b.dataset.view === view);
  });
  if (view === "map") {
    setTimeout(() => { if (map) map.invalidateSize(); }, 60);
  } else if (typeof onViewChange === "function") {
    setTimeout(() => onViewChange("analysis"), 60);
  }
}

function edgeLatLng(lat, lon, radius_m) {
  const dlon = (radius_m / 111320) / Math.max(0.2, Math.cos((lat * Math.PI) / 180));
  return L.latLng(lat, lon + dlon);
}

export function initAmap() {
  if (map) return;
  map = L.map("amap", {
    center: [state.center.lat, state.center.lon],
    zoom: 11,
    zoomControl: true,
    attributionControl: true,
  });
  L.tileLayer(
    "https://webrd02.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}",
    { maxZoom: 18, attribution: "© 高德" },
  ).addTo(map);

  map.on("dblclick", (e) => setCenter(e.latlng.lat, e.latlng.lng, false));
  map.doubleClickZoom.disable();

  setCenter(state.center.lat, state.center.lon, false);
  setTimeout(() => map.invalidateSize(), 100);
}

export function getMap() {
  return map;
}

export function setCenter(lat, lon, panTo = true) {
  state.center = { lat, lon };
  ensureAoiLayers();
  if (centerMarker) centerMarker.setLatLng([lat, lon]);
  if (bboxCircle) bboxCircle.setLatLng([lat, lon]);
  syncHandle();
  updateAoiHud();
  if (panTo && map) map.panTo([lat, lon]);
}

export function setRadiusKm(r, fromSlider = false) {
  state.radius_km = Math.round(r * 10) / 10;
  if (!fromSlider && $("radius-slider")) {
    $("radius-slider").value = String(state.radius_km);
  }
  if ($("radius-value")) {
    $("radius-value").textContent = state.radius_km.toFixed(1) + " km";
  }
  if (bboxCircle) {
    bboxCircle.setRadius(state.radius_km * 1000);
    const col = qualityColor(radiusQuality(state.radius_km));
    bboxCircle.setStyle({ color: col, fillColor: col });
  }
  syncHandle();
  updateAoiHud();
}

function ensureAoiLayers() {
  if (!map) return;
  if (!centerMarker) {
    const icon = L.divIcon({
      className: "",
      html: '<div style="width:16px;height:16px;background:#c0392b;border:2px solid #fff;border-radius:50%;box-shadow:0 0 0 1px #c0392b;cursor:move"></div>',
      iconSize: [16, 16],
      iconAnchor: [8, 8],
    });
    centerMarker = L.marker([state.center.lat, state.center.lon], {
      icon, draggable: true, zIndexOffset: 1000,
    }).addTo(map);
    centerMarker.on("drag", (e) => {
      const ll = e.target.getLatLng();
      state.center = { lat: ll.lat, lon: ll.lng };
      if (bboxCircle) bboxCircle.setLatLng(ll);
      syncHandle();
      updateAoiHud();
    });
  }
  if (!bboxCircle) {
    const col = qualityColor(radiusQuality(state.radius_km));
    bboxCircle = L.circle([state.center.lat, state.center.lon], {
      radius: state.radius_km * 1000,
      color: col, weight: 2, opacity: 0.85,
      fillColor: col, fillOpacity: 0.08,
      dashArray: "6,4",
    }).addTo(map);
  }
  if (!handleMarker) {
    const hIcon = L.divIcon({
      className: "",
      html: '<div style="width:14px;height:14px;background:#fff;border:2px solid #1a1a1a;border-radius:50%;box-shadow:0 1px 3px rgba(0,0,0,.25);cursor:ew-resize" title="拖动改半径"></div>',
      iconSize: [14, 14],
      iconAnchor: [7, 7],
    });
    handleMarker = L.marker(
      edgeLatLng(state.center.lat, state.center.lon, state.radius_km * 1000),
      { icon: hIcon, draggable: true, zIndexOffset: 1100 },
    ).addTo(map);
    handleMarker.on("dragstart", () => { map.dragging.disable(); });
    handleMarker.on("drag", (e) => {
      const c = L.latLng(state.center.lat, state.center.lon);
      let km = c.distanceTo(e.target.getLatLng()) / 1000;
      km = Math.min(state.aoiLimits.max_radius_km, Math.max(state.aoiLimits.min_radius_km * 0.5, km));
      setRadiusKm(km);
    });
    handleMarker.on("dragend", () => {
      map.dragging.enable();
      if (radiusQuality(state.radius_km) === "invalid") {
        setRadiusKm(clampRadius(state.radius_km));
      }
    });
  }
}

function syncHandle() {
  if (!handleMarker || !state.center) return;
  handleMarker.setLatLng(
    edgeLatLng(state.center.lat, state.center.lon, state.radius_km * 1000),
  );
}

export function markAnalysisReady() {
  state.hasAnalysis = true;
  const tab = $("tab-analysis");
  if (tab) tab.disabled = false;
  switchView("analysis");
}

export function wireAoiControls() {
  $("mode-tabs")?.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-view]");
    if (!btn || btn.disabled) return;
    switchView(btn.dataset.view);
  });
  $("radius-slider")?.addEventListener("input", (e) => {
    setRadiusKm(parseFloat(e.target.value), true);
  });
}
