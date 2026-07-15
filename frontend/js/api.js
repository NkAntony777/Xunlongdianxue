/** 后端 API 封装 */
import { API } from "./config.js";
import { base64ToBytes } from "./utils.js";

export async function fetchAoiLimits() {
  const resp = await fetch(`${API}/api/aoi/limits`);
  if (!resp.ok) throw new Error("无法获取 AOI 约束");
  return resp.json();
}

export async function searchLocation(query, limit = 6) {
  const resp = await fetch(`${API}/api/location/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, limit, countrycodes: "cn" }),
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.detail || "搜索失败");
  return data.results || [];
}

export async function fetchElevation(lon, lat, radius_km) {
  const resp = await fetch(`${API}/api/elevation/fetch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lon, lat, radius_km }),
  });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.detail || resp.statusText);
  return data;
}

export async function fetchWater(lon, lat, radius_km) {
  const resp = await fetch(`${API}/api/water/fetch`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    // 固定 4326 落盘，避免投影坐标被当成经纬度
    body: JSON.stringify({ lon, lat, radius_km, target_crs: "EPSG:4326" }),
  });
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    return {
      ok: false,
      features: [],
      count: 0,
      degraded: true,
      warning: String(err.detail || resp.statusText || "水系服务不可用"),
    };
  }
  const data = await resp.json();
  // 后端 ok 字段优先；缺省时有要素即视为成功
  const ok = data.ok !== undefined ? Boolean(data.ok) : (Number(data.count) > 0 || !data.degraded);
  return { ok, ...data };
}

export async function saveTiffBytes(bytes) {
  const resp = await fetch(`${API}/api/cache/save_tmp`, {
    method: "POST",
    headers: { "Content-Type": "application/octet-stream" },
    body: bytes,
  });
  if (!resp.ok) throw new Error("保存 DEM 临时文件失败");
  const j = await resp.json();
  return j.path;
}

export async function saveGeoJsonText(text) {
  const resp = await fetch(`${API}/api/cache/save_text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ content: text }),
  });
  if (!resp.ok) throw new Error("保存水系临时文件失败");
  const j = await resp.json();
  return j.path;
}

export async function fetchLayers(demPath, waterPath, basemapMode = "elevation") {
  const q = new URLSearchParams({
    dem_path: demPath,
    water_path: waterPath || "",
    top_k: "10",
    mode: basemapMode,
    sample_step: "3",
    contour_interval: "30",
    influence_buffer_m: "1100",
  });
  const resp = await fetch(`${API}/api/layers/all?${q}`);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    console.warn("layers/all failed", demPath, err);
    return null;
  }
  return resp.json();
}

/** 指定穴心 (x,y) 计算四象；用于候选穴点选 */
export async function fetchFourBeastsAt(demPath, waterPath, x, y) {
  const q = new URLSearchParams({
    dem_path: demPath,
    water_path: waterPath || "",
    center_x: String(x),
    center_y: String(y),
  });
  const resp = await fetch(`${API}/api/layers/four-beasts?${q}`);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({}));
    throw new Error(err.detail || "四象计算失败");
  }
  return resp.json();
}

export function geotiffBase64ToBytes(b64) {
  return base64ToBytes(b64);
}
