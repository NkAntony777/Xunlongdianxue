/**
 * 前端轻量测试（Node ESM，无浏览器依赖）
 * 运行: node frontend/tests/run.mjs
 */
import { readFileSync, readdirSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

import {
  isGeographicCrs,
  isWebMercator,
  mercatorToLonLat,
  lonLatToMercator,
  resolveLonLatBounds,
  worldToLonLat,
  LANGZHONG_BBOX_LONLAT,
  MERC_R,
} from "../js/geo.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FRONTEND = join(__dirname, "..");
const ROOT = join(FRONTEND, "..");

let passed = 0;
let failed = 0;

function assert(cond, msg) {
  if (cond) {
    passed += 1;
    return;
  }
  failed += 1;
  console.error("  FAIL:", msg);
}

function almostEqual(a, b, eps = 1e-6) {
  return Math.abs(a - b) <= eps;
}

console.log("== geo.js ==");

assert(isGeographicCrs("EPSG:4326"), "4326 is geographic");
assert(isGeographicCrs("WGS 84"), "WGS is geographic");
assert(!isGeographicCrs("EPSG:3857"), "3857 not geographic");
assert(isWebMercator("EPSG:3857"), "3857 is mercator");
assert(isWebMercator("EPSG:900913"), "900913 is mercator");
assert(!isWebMercator("EPSG:4326"), "4326 not mercator");

{
  const { lon, lat } = mercatorToLonLat(0, 0);
  assert(almostEqual(lon, 0) && almostEqual(lat, 0), "mercator origin → 0,0");
}
{
  // 约东经 105.97, 北纬 31.58 的 Web Mercator
  const m = lonLatToMercator(105.97, 31.58);
  const back = mercatorToLonLat(m.x, m.y);
  assert(
    almostEqual(back.lon, 105.97, 1e-5) && almostEqual(back.lat, 31.58, 1e-5),
    "lonLat ↔ mercator roundtrip ~阆中",
  );
  assert(Math.abs(m.x) > 1e6, "mercator x 量级正确");
}

{
  const b = resolveLonLatBounds({
    bboxLonLat: [105.9, 31.5, 106.0, 31.6],
  });
  assert(
    b && almostEqual(b[0], 105.9) && almostEqual(b[3], 31.6),
    "优先 bboxLonLat",
  );
}

{
  const b = resolveLonLatBounds({
    bbox: [105.85, 31.5, 106.0, 31.65],
    crs: "EPSG:4326",
  });
  assert(b && almostEqual(b[0], 105.85), "地理 CRS 直接用 bbox");
}

{
  const sw = lonLatToMercator(105.85, 31.5);
  const ne = lonLatToMercator(106.0, 31.65);
  const b = resolveLonLatBounds({
    bbox: [sw.x, sw.y, ne.x, ne.y],
    crs: "EPSG:3857",
  });
  assert(
    b
      && almostEqual(b[0], 105.85, 1e-4)
      && almostEqual(b[1], 31.5, 1e-4)
      && almostEqual(b[2], 106.0, 1e-4)
      && almostEqual(b[3], 31.65, 1e-4),
    "3857 bbox → lonlat",
  );
}

{
  const b = resolveLonLatBounds({ mode: "demo" });
  assert(
    b
      && almostEqual(b[0], LANGZHONG_BBOX_LONLAT[0])
      && almostEqual(b[2], LANGZHONG_BBOX_LONLAT[2]),
    "demo 回退阆中",
  );
}

{
  const bbox = [0, 0, 100, 50];
  const bboxLonLat = [100, 30, 101, 31];
  const mid = worldToLonLat(50, 25, { bbox, bboxLonLat });
  assert(
    mid
      && almostEqual(mid.lon, 100.5)
      && almostEqual(mid.lat, 30.5),
    "线性 world→lonlat 中点",
  );
  assert(worldToLonLat(null, 1, { bbox, bboxLonLat }) === null, "无效坐标 null");
}

assert(Number.isFinite(MERC_R) && MERC_R > 6e6, "MERC_R 常量");

console.log("== utils 纯函数（内联校验） ==");
{
  // 与 utils.js 一致的最小实现，避免 DOM
  function fmt(v, n) {
    if (v === null || v === undefined || Number.isNaN(+v)) return "—";
    return Number(v).toFixed(n);
  }
  function esc(s) {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;")
      .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }
  assert(fmt(1.234, 2) === "1.23", "fmt fixed");
  assert(fmt(null, 1) === "—", "fmt null");
  assert(esc('<a "b">') === "&lt;a &quot;b&quot;&gt;", "esc html");
}

console.log("== JS 语法检查 (node --check) ==");
{
  const jsDir = join(FRONTEND, "js");
  const files = readdirSync(jsDir).filter((f) => f.endsWith(".js"));
  for (const f of files) {
    const p = join(jsDir, f);
    const r = spawnSync(process.execPath, ["--check", p], { encoding: "utf8" });
    assert(r.status === 0, `syntax ${f}${r.stderr ? ": " + r.stderr.trim() : ""}`);
  }
  assert(files.length >= 10, `js 模块数量 ${files.length}`);
}

console.log("== 页面骨架与接线 ==");
{
  const html = readFileSync(join(FRONTEND, "index.html"), "utf8");
  for (const id of [
    "amap", "result-amap", "display-modes", "layer-stack",
    "analysis-toggles", "basemap-modes", "mode-tabs", "btn-analyze",
  ]) {
    assert(html.includes(`id="${id}"`) || html.includes(`id='${id}'`), `html#${id}`);
  }
  assert(html.includes('data-display="gaode"'), "高德对照按钮");
  assert(html.includes('data-display="analysis"'), "分析图按钮");
  assert(html.includes("/static/js/app.js") || html.includes("js/app.js"), "入口 app.js");

  const app = readFileSync(join(FRONTEND, "js", "app.js"), "utf8");
  assert(app.includes("wireResultMap"), "app 接线 result-map");
  assert(app.includes("wireAnalysisMap"), "app 接线 analysis-map");

  const rm = readFileSync(join(FRONTEND, "js", "result-map.js"), "utf8");
  assert(rm.includes('from "./geo.js"'), "result-map 使用 geo.js");
  assert(rm.includes("setCandidateSelectHandler"), "候选点选回调");
}

console.log("== 配置与 state ==");
{
  const stateSrc = readFileSync(join(FRONTEND, "js", "state.js"), "utf8");
  assert(stateSrc.includes("displayMode"), "state.displayMode");
  assert(stateSrc.includes("layerImages"), "state.layerImages");
  assert(stateSrc.includes("bboxLonLat"), "state.bboxLonLat");
}

console.log("");
console.log(`frontend tests: ${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
console.log("OK");
