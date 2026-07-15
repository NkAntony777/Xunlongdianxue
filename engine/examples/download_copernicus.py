"""从 Copernicus DEM AWS 公有桶下载阆中区域 DEM。

数据源：https://registry.opendata.aws/copernicus-dem/
阆中 (105.85-106.00E, 31.50-31.65N) → tile N31_E105
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.errors import RasterioIOError

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
LANZHONG_BBOX = (105.85, 31.50, 106.00, 31.65)


def _tile_name(lon: float, lat: float) -> tuple[str, str]:
    """返回 (lat_tile, lon_tile) 命名部分。如 (N31, E105)"""
    lat_str = f"N{int(lat):02d}" if lat >= 0 else f"S{abs(int(lat)):02d}"
    lon_str = f"E{int(lon):03d}" if lon >= 0 else f"W{abs(int(lon)):03d}"
    return lat_str, lon_str


def _tile_url(lon: float, lat: float) -> str:
    lt, lz = _tile_name(lon, lat)
    return (f"{BUCKET}/Copernicus_DSM_COG_10_{lt}_00_{lz}_00_DEM/"
            f"Copernicus_DSM_COG_10_{lt}_00_{lz}_00_DEM.tif")


def download_copernicus(bbox: tuple = None) -> Path:
    if bbox is None:
        bbox = LANZHONG_BBOX

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / "langzhong_cop30.tif"

    minx, miny, maxx, maxy = bbox
    # 计算需要的 tile（取中心点所在 tile）
    cx = (minx + maxx) / 2
    cy = (miny + maxy) / 2
    url = _tile_url(cx, cy)

    print(f"[下载] 读取 tile: {url.split('/')[-2]}")
    print(f"       范围: {minx:.3f}E - {maxx:.3f}E, {miny:.3f}N - {maxy:.3f}N")

    try:
        with rasterio.open(url) as src:
            print(f"       tile 大小: {src.width}x{src.height}, CRS={src.crs}")
            window = from_bounds(minx, miny, maxx, maxy, src.transform)
            aff = src.window_transform(window)
            data = src.read(1, window=window).astype("float64")

            # 处理 NoData
            data[data < -1000] = np.nan

            print(f"       裁剪: {data.shape[1]}x{data.shape[0]} 像素")
            print(f"       高程: {np.nanmin(data):.0f} - {np.nanmax(data):.0f} m")

            profile = src.profile.copy()
            profile.update({
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": aff,
                "dtype": "float64",
                "compress": "lzw",
            })
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data, 1)
            print(f"[完成] {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")
            return out_path
    except RasterioIOError as e:
        print(f"[错误] tile 读取失败: {e}")
        print()
        print("备选方案:")
        print("  1. USGS EarthExplorer (免费, 需注册)")
        print("     https://earthexplorer.usgs.gov/")
        print("     → 数据集: 'Copernicus GLO-30' 或 'SRTM 1-Arc Second Global'")
        print("     → bbox: 105.85-106.00E, 31.50-31.65N")
        print()
        print("  2. OpenTopography (免费, 需注册 API key)")
        print("     https://opentopography.org/ → 注册 → My Account → API Keys")
        print(f"     engine\\.venv\\Scripts\\python.exe -m engine.examples.prepare_data opentopo --api-key <你的key>")
        print()
        print("  3. 地理空间数据云 (中国, 免费注册)")
        print("     https://www.gscloud.cn/ → 搜索 'GDEM V3'")
        raise


if __name__ == "__main__":
    print("=" * 60)
    print("  Copernicus DEM 30m - 阆中区域下载")
    print("=" * 60)
    out = download_copernicus()
    print(f"\n[使用] engine\\.venv\\Scripts\\python.exe -m engine.cli"
          f" --dem {out} --water data\\langzhong_rivers.geojson --dragon-vein --top-k 5")
