/**
 * 寻龙点穴引擎 — 前端入口
 * 职责：组装模块、绑定事件、启动
 */
import { state } from "./state.js";
import { $, setStatus } from "./utils.js";
import {
  loadAoiLimits, initAmap, setRadiusKm, updateAoiHud,
  switchView, wireAoiControls, setViewChangeHandler,
} from "./aoi.js";
import { wireAnalysisMap, onAnalysisViewShown } from "./analysis-map.js";
import { wireLayerToggles, selectCandidate } from "./render-ui.js";
import { wireAnalysisButtons, reloadBasemap } from "./analysis.js";
import { wireSearch } from "./search.js";
import {
  wireResultMap, onResultMapShown, setCandidateSelectHandler,
} from "./result-map.js";

async function boot() {
  setViewChangeHandler((view) => {
    if (view === "analysis") {
      onAnalysisViewShown();
      onResultMapShown();
    }
  });
  wireLayerToggles.onBasemapChange = () => reloadBasemap();
  setCandidateSelectHandler((id) => selectCandidate(id));

  wireAoiControls();
  wireAnalysisMap();
  wireResultMap();
  wireLayerToggles();
  wireAnalysisButtons();
  wireSearch();

  await loadAoiLimits();
  state.radius_km = state.aoiLimits.default_radius_km || 8;
  if ($("radius-slider")) $("radius-slider").value = String(state.radius_km);
  if ($("radius-value")) {
    $("radius-value").textContent = state.radius_km.toFixed(1) + " km";
  }

  initAmap();
  setRadiusKm(state.radius_km);
  updateAoiHud();
  // 分析视图默认「分析图」模式 class
  $("analysis-view")?.classList.add("mode-analysis");
  switchView("map");
  setStatus("在高德图上圈选分析区（推荐 5–15 km），然后点「开始分析」");
  if ($("page-sub")) {
    $("page-sub").textContent =
      "高德底图自由缩放 · 拖动中心/边缘圈定范围 · 拉取 ESRI DEM + OSM 水系后分析。";
  }
}

boot().catch((e) => {
  console.error("boot failed", e);
  setStatus("启动失败: " + e.message, "err");
});
