/** 应用常量与默认路径 */
export const API = "";
export const DEFAULT_DEM = "data/langzhong_cop30.tif";
export const DEFAULT_WATER = "data/langzhong_rivers_osm.geojson";
export const FALLBACK_DEM = "data/langzhong_dem.tif";
export const FALLBACK_WATER = "data/langzhong_rivers.geojson";

export const BEAST_NAMES = {
  shaozu: "少祖", xuanwu: "玄武", zhuque: "朱雀",
  qinglong: "青龙", baihu: "白虎",
};
/** 角色短名 / 符号（图例） */
export const BEAST_SHORT = {
  shaozu: "祖", xuanwu: "玄", zhuque: "雀",
  qinglong: "龙", baihu: "虎",
};
/** 主色 */
export const BEAST_COLORS = {
  shaozu: "#475569",
  xuanwu: "#0f172a",
  zhuque: "#dc2626",
  qinglong: "#0d9488",
  baihu: "#d97706",
};
export const BEAST_RING = {
  shaozu: "#94a3b8",
  xuanwu: "#f87171",
  zhuque: "#fca5a5",
  qinglong: "#5eead4",
  baihu: "#fcd34d",
};
/** 柔和底色（标签板） */
export const BEAST_SOFT = {
  shaozu: "rgba(248,250,252,0.94)",
  xuanwu: "rgba(15,23,42,0.92)",
  zhuque: "rgba(254,242,242,0.95)",
  qinglong: "rgba(240,253,250,0.95)",
  baihu: "rgba(255,251,235,0.95)",
};
export const BEAST_ORDER = ["shaozu", "xuanwu", "zhuque", "qinglong", "baihu"];

export const FACING_METHOD_LABEL = {
  face_water: "面水（最近水系，已与靠山校验）",
  back_to_high_terrain: "背靠高地（地形推断）",
  back_high_face_water: "背山面水（靠山+前方得水）",
  mingtang_face_water: "明堂面水（开阔+得水）",
  back_high_over_nearest_water: "背靠高地（否决最近岸反局）",
  sit_to_dragon_source: "坐靠来龙（向龙源）",
  dragon_sit_face_water: "坐靠来龙·面水有情",
  dragon_source_sit: "坐靠来龙（祖峰定坐）",
  default_south: "默认坐北朝南",
  user_facing: "用户指定朝向",
  user_override: "用户指定朝向",
};
