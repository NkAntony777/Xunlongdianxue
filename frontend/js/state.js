/** 全局可变状态（单例） */
import { DEFAULT_DEM, DEFAULT_WATER } from "./config.js";

export const state = {
  mode: "map",
  view: "map",
  /** analysis = DEM 分析图叠层；gaode = 高德底图 + 穴点/热力 */
  displayMode: "analysis",
  center: { lat: 31.58, lon: 105.97 },
  radius_km: 8,
  aoiLimits: {
    min_radius_km: 3, max_radius_km: 25,
    recommended_min_km: 5, recommended_max_km: 15,
    default_radius_km: 8,
  },
  basemapMode: "elevation",
  layerVisible: {
    score: true, contours: true, water: true, influence: true,
    buildings: true, beasts: true, candidates: false, ridges: false, peak: true,
  },
  dem: null,
  bbox: null,
  bboxLonLat: null,
  /** 分析图层 data URL，供高德 imageOverlay 复用 */
  layerImages: {
    basemap: null,
    score: null,
    contours: null,
    water: null,
    influence: null,
    buildings: null,
  },
  fb: null,
  /** 场评最高点对应的四象快照（点选候选后可还原） */
  fbPeak: null,
  centerXY: null,
  centerXYPeak: null,
  facing: 0,
  sit: 180,
  facingMethod: "",
  fbMeta: null,
  candidates: [],
  selectedCand: null,
  /** peak | candidate | loading */
  beastsSource: "peak",
  beastsLoading: false,
  /** candId → 四象 API 结果缓存 */
  fbCache: {},
  ridges: [],
  peakXY: null,
  demPath: DEFAULT_DEM,
  waterPath: DEFAULT_WATER,
  placeName: "阆中",
  hasAnalysis: false,
};
