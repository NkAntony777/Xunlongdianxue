/** 分析流程：本地 Demo / 在线圈选分析 */
import {
  DEFAULT_DEM, DEFAULT_WATER, FALLBACK_DEM, FALLBACK_WATER,
} from "./config.js";
import { state } from "./state.js";
import { $, setStatus, sleep } from "./utils.js";
import {
  startProgress, setProgress, finishProgress, relabelStep, showLoading,
} from "./progress.js";
import {
  fetchElevation, fetchWater, fetchLayers,
  saveTiffBytes, saveGeoJsonText, geotiffBase64ToBytes,
} from "./api.js";
import {
  radiusQuality, qualityLabel, updateAoiHud, markAnalysisReady, switchView,
  setCenter, setRadiusKm,
} from "./aoi.js";
import { applyAll } from "./render-ui.js";

export async function loadLocalDemo() {
  startProgress("加载阆中 Demo", [
    "读取本地 DEM",
    "读取本地水系",
    "地形 / 四象 / 候选穴分析",
    "渲染图层",
  ]);
  setStatus("渲染本地 DEM…");
  const t0 = performance.now();
  try {
    setProgress(0, "active", "加载 COP30 / 备用 DEM…");
    let demPath = state.demPath || DEFAULT_DEM;
    let waterPath = state.waterPath || DEFAULT_WATER;
    setProgress(0, "done");
    setProgress(1, "active");
    setProgress(1, "done", waterPath);
    setProgress(2, "active", "分析中，约需数十秒…");
    let layers = await fetchLayers(demPath, waterPath, state.basemapMode);
    if (!layers) {
      demPath = FALLBACK_DEM;
      layers = await fetchLayers(demPath, waterPath, state.basemapMode);
    }
    if (!layers) {
      waterPath = FALLBACK_WATER;
      layers = await fetchLayers(demPath, waterPath, state.basemapMode);
    }
    if (!layers) throw new Error("无法加载本地 Demo 数据");
    setProgress(2, "done");
    setProgress(3, "active", "叠合图层…");
    state.demPath = demPath;
    state.waterPath = waterPath;
    state.mode = "demo";
    applyAll(layers);
    markAnalysisReady();
    setProgress(3, "done");
    const sec = ((performance.now() - t0) / 1000).toFixed(1);
    setStatus(`Demo 就绪 · ${sec}s · 候选 ${state.candidates.length}`, "ok");
    finishProgress(true, "Demo 加载完成");
  } catch (e) {
    console.error(e);
    setStatus("错误: " + e.message, "err");
    if ($("dem-info")) $("dem-info").innerHTML = `<div class="empty">加载失败</div>`;
    finishProgress(false, e.message);
  }
}

export async function runOnlineAnalysis() {
  if (!state.center) return;
  const q = radiusQuality(state.radius_km);
  if (q === "invalid") {
    setStatus(
      `圈选无效：半径须在 ${state.aoiLimits.min_radius_km}–${state.aoiLimits.max_radius_km} km`,
      "err",
    );
    switchView("map");
    return;
  }
  if ($("btn-analyze")) $("btn-analyze").disabled = true;
  const t0 = performance.now();
  let waterWarning = "";
  startProgress("正在分析选区", [
    "校验分析范围",
    "拉取高程 DEM（ESRI）",
    "拉取水系（OSM Overpass）",
    "写入临时数据",
    "地形 / 四象 / 候选穴分析",
    "渲染图层与标注",
  ]);
  try {
    setProgress(0, "active", `半径 ${state.radius_km.toFixed(1)} km · 校验中`);
    setStatus(`准备分析 r=${state.radius_km.toFixed(1)} km…`);
    await sleep(120);
    setProgress(0, "done");

    setProgress(1, "active", "连接 ESRI World Elevation…");
    setStatus("拉取 DEM…");
    const elev = await fetchElevation(
      state.center.lon, state.center.lat, state.radius_km,
    );
    state.bboxLonLat = elev.bbox_lonlat || null;
    setProgress(
      1, "done",
      `DEM ${elev.shape?.[1]}×${elev.shape?.[0]} · ${Math.round(elev.elevation_min || 0)}–${Math.round(elev.elevation_max || 0)} m`,
    );

    setProgress(2, "active", "连接 OSM Overpass（多节点重试）…");
    setStatus("拉取水系…");
    const water = await fetchWater(
      state.center.lon, state.center.lat, state.radius_km,
    );
    if (!water.ok) {
      waterWarning = water.warning || "水系服务不可用";
      relabelStep(2, "拉取水系（已跳过，服务不可用）");
      setProgress(2, "skip", "水系跳过，将仅用 DEM 分析");
    } else if (water.warning || water.degraded) {
      waterWarning = water.warning || "水系降级";
      relabelStep(2, `拉取水系（${water.count || 0} 条，有警告）`);
      setProgress(2, "done", waterWarning);
    } else {
      relabelStep(2, `拉取水系（${water.count || 0} 条）`);
      setProgress(2, "done", water.count ? `已获取 ${water.count} 条水系` : "范围内无水系要素");
    }

    setProgress(3, "active", "保存 GeoTIFF / GeoJSON…");
    setStatus("保存临时文件…");
    const tifPath = await saveTiffBytes(geotiffBase64ToBytes(elev.geotiff_base64));
    const waterPath = await saveGeoJsonText(JSON.stringify({
      type: "FeatureCollection",
      features: water.features || [],
    }));
    setProgress(3, "done");

    setProgress(4, "active", "计算坡度、评分、四象与候选穴…");
    setStatus("地形与四象分析…");
    const layers = await fetchLayers(tifPath, waterPath, state.basemapMode);
    if (!layers) throw new Error("图层渲染失败");
    setProgress(4, "done");

    setProgress(5, "active", "叠合图层与标注…");
    state.mode = "online";
    state.demPath = tifPath;
    state.waterPath = waterPath;
    applyAll(layers, elev);
    markAnalysisReady();
    setProgress(5, "done");

    const sec = ((performance.now() - t0) / 1000).toFixed(1);
    const qnote = elev.radius_quality && elev.radius_quality !== "ok"
      ? ` · 尺度${qualityLabel(elev.radius_quality)}` : "";
    const wnote = waterWarning ? " · 水系降级" : "";
    setStatus(`分析完成 · ${sec}s · 候选 ${state.candidates.length}${qnote}${wnote}`, "ok");
    if ($("page-title")) {
      $("page-title").textContent = `${state.placeName || "自定义区域"} 地形寻龙`;
    }
    if ($("page-sub")) {
      $("page-sub").textContent =
        `在线分析 · 中心 ${state.center.lat.toFixed(4)},${state.center.lon.toFixed(4)} · 半径 ${state.radius_km.toFixed(1)} km` +
        (waterWarning ? " · 水系数据不完整" : "");
    }
    finishProgress(
      true,
      waterWarning ? `完成（水系：${waterWarning.slice(0, 40)}）` : "全部步骤完成",
    );
  } catch (e) {
    console.error(e);
    setStatus("错误: " + e.message, "err");
    finishProgress(false, e.message);
    switchView("map");
  } finally {
    updateAoiHud();
  }
}

export async function reloadBasemap() {
  if (state.mode === "demo") {
    await loadLocalDemo();
    return;
  }
  if (!state.demPath) return;
  showLoading(true, "切换底图…");
  try {
    const layers = await fetchLayers(state.demPath, state.waterPath, state.basemapMode);
    if (layers) applyAll(layers);
  } finally {
    showLoading(false);
  }
}

export function wireAnalysisButtons() {
  $("btn-demo")?.addEventListener("click", () => {
    state.mode = "demo";
    state.placeName = "阆中";
    state.demPath = DEFAULT_DEM;
    state.waterPath = DEFAULT_WATER;
    setCenter(31.58, 105.97, true);
    setRadiusKm(state.aoiLimits.default_radius_km || 8);
    if ($("page-title")) $("page-title").textContent = "阆中真实地形寻龙 Demo";
    if ($("page-sub")) {
      $("page-sub").textContent = "本地 COP30 DEM + OSM 水系 · 四象主轴由算法实时计算。";
    }
    loadLocalDemo();
  });
  $("btn-analyze")?.addEventListener("click", () => runOnlineAnalysis());
  $("btn-3d")?.addEventListener("click", () => {
    alert("三维可旋转地形视图即将上线（three.js + DEM 纹理）。当前请使用 2D 分析场。");
  });
}
