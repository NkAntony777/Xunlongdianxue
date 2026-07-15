/**
 * 纯地理工具（无 DOM / Leaflet 依赖，便于 Node 单测）
 */

/** Web Mercator 地球半径 (m) */
export const MERC_R = 6378137;

/** 阆中 Demo 默认经纬度包围盒 [minLon, minLat, maxLon, maxLat] */
export const LANGZHONG_BBOX_LONLAT = [105.85, 31.5, 106.0, 31.65];

/** 是否地理 CRS（经纬度） */
export function isGeographicCrs(crs) {
  if (!crs) return false;
  const s = String(crs).toUpperCase();
  return s.includes("4326") || s.includes("GEOG") || s.includes("WGS");
}

/** 是否 Web Mercator */
export function isWebMercator(crs) {
  if (!crs) return false;
  const s = String(crs).toUpperCase();
  return (
    s.includes("3857")
    || s.includes("900913")
    || s.includes("WEB_MERCATOR")
    || s.includes("PSEUDO-MERCATOR")
  );
}

export function mercatorToLonLat(x, y) {
  const lon = (x / MERC_R) * (180 / Math.PI);
  const lat = (2 * Math.atan(Math.exp(y / MERC_R)) - Math.PI / 2) * (180 / Math.PI);
  return { lon, lat };
}

export function lonLatToMercator(lon, lat) {
  const x = (lon * Math.PI) / 180 * MERC_R;
  const y = Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360)) * MERC_R;
  return { x, y };
}

/**
 * 解析有效经纬度包围盒 [minLon, minLat, maxLon, maxLat]
 * @param {{ bboxLonLat?: number[]|null, bbox?: number[]|null, crs?: string|null, mode?: string }} ctx
 */
export function resolveLonLatBounds(ctx = {}) {
  const { bboxLonLat, bbox, crs, mode } = ctx;
  if (bboxLonLat && bboxLonLat.length === 4) {
    const [a, b, c, d] = bboxLonLat.map(Number);
    if ([a, b, c, d].every(Number.isFinite) && c > a && d > b) {
      return [a, b, c, d];
    }
  }
  if (bbox && bbox.length === 4) {
    const [minx, miny, maxx, maxy] = bbox.map(Number);
    if (isGeographicCrs(crs) || (
      Math.abs(minx) <= 180 && Math.abs(maxx) <= 180
      && Math.abs(miny) <= 90 && Math.abs(maxy) <= 90
      && (maxx - minx) < 5 && (maxy - miny) < 5
    )) {
      return [minx, miny, maxx, maxy];
    }
    if (isWebMercator(crs) || (Math.abs(minx) > 180 || Math.abs(maxx) > 180)) {
      const sw = mercatorToLonLat(minx, miny);
      const ne = mercatorToLonLat(maxx, maxy);
      return [sw.lon, sw.lat, ne.lon, ne.lat];
    }
  }
  if (mode === "demo" || !bboxLonLat) {
    return [...LANGZHONG_BBOX_LONLAT];
  }
  return null;
}

/**
 * 分析世界坐标 → WGS84
 * @param {number} x
 * @param {number} y
 * @param {{ bboxLonLat?: number[]|null, bbox?: number[]|null, crs?: string|null, mode?: string }} ctx
 */
export function worldToLonLat(x, y, ctx = {}) {
  if (x == null || y == null || !Number.isFinite(x) || !Number.isFinite(y)) {
    return null;
  }
  const llb = resolveLonLatBounds(ctx);
  const bbox = ctx.bbox;
  if (llb && bbox && bbox.length === 4) {
    const [minx, miny, maxx, maxy] = bbox.map(Number);
    const [minLon, minLat, maxLon, maxLat] = llb;
    const dx = maxx - minx;
    const dy = maxy - miny;
    if (Math.abs(dx) > 1e-12 && Math.abs(dy) > 1e-12) {
      const lon = minLon + ((x - minx) / dx) * (maxLon - minLon);
      const lat = minLat + ((y - miny) / dy) * (maxLat - minLat);
      if (Number.isFinite(lon) && Number.isFinite(lat)) return { lon, lat };
    }
  }
  const crs = ctx.crs;
  if (isGeographicCrs(crs) || (Math.abs(x) <= 180 && Math.abs(y) <= 90)) {
    return { lon: x, lat: y };
  }
  if (isWebMercator(crs) || Math.abs(x) > 180) {
    return mercatorToLonLat(x, y);
  }
  return null;
}
