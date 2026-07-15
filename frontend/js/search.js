/** 地点搜索 */
import { state } from "./state.js";
import { $, setStatus, esc } from "./utils.js";
import { searchLocation } from "./api.js";
import { initAmap, setCenter, switchView, getMap } from "./aoi.js";

let searchTimer = null;

export function wireSearch() {
  $("search-input")?.addEventListener("input", (e) => {
    const q = e.target.value.trim();
    clearTimeout(searchTimer);
    if (q.length < 2) {
      if ($("search-results")) $("search-results").style.display = "none";
      return;
    }
    searchTimer = setTimeout(() => doSearch(q), 350);
  });
  document.addEventListener("click", (e) => {
    if (!e.target.closest("#search-box") && $("search-results")) {
      $("search-results").style.display = "none";
    }
  });
}

async function doSearch(query) {
  try {
    const results = await searchLocation(query, 6);
    renderSearchResults(results);
  } catch (e) {
    setStatus("搜索失败: " + e.message, "err");
  }
}

function renderSearchResults(results) {
  const box = $("search-results");
  if (!box) return;
  if (!results.length) { box.style.display = "none"; return; }
  box.innerHTML = results.map((r, i) => `
    <div class="item" data-idx="${i}">
      <div class="name">${esc(r.short_name || r.name)}</div>
      <div class="meta">${r.lat.toFixed(4)}, ${r.lon.toFixed(4)}</div>
    </div>`).join("");
  box.style.display = "block";
  box.querySelectorAll(".item").forEach((el) => {
    el.addEventListener("click", () => {
      const r = results[+el.dataset.idx];
      if ($("search-input")) $("search-input").value = r.short_name || r.name;
      box.style.display = "none";
      state.placeName = r.short_name || r.name;
      state.mode = "online";
      initAmap();
      setCenter(r.lat, r.lon, true);
      const map = getMap();
      if (map) map.setView([r.lat, r.lon], Math.max(11, map.getZoom()));
      switchView("map");
      if ($("page-title")) $("page-title").textContent = `${state.placeName} 地形寻龙`;
      if ($("page-sub")) {
        $("page-sub").textContent = "在高德图上调整分析圈 →「开始分析」拉取 ESRI DEM + OSM 水系。";
      }
      setStatus(`已定位 ${state.placeName}，请确认圈选范围后分析`);
    });
  });
}
