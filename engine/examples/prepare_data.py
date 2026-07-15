"""下载真实 DEM：支持 OpenTopography / 地理空间数据云 / 手动指定。

用法：
  python prepare_data.py opentopo              # 从 OpenTopography 下载
  python prepare_data.py opentopo --api-key xxx # 使用自己的 key
  python prepare_data.py manual                  # 手动指定 DEM 路径
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# 阆中研究区 bbox
LANZHONG_BBOX = {
    "south": 31.50,
    "north": 31.65,
    "west": 105.85,
    "east": 106.00,
}


def download_bmi_topography(
    api_key: str,
    dem_type: str = "COP30",
    bbox: dict[str, float] | None = None,
) -> Path:
    """使用 bmi-topography 库下载 DEM。

    支持的数据集：
      - SRTMGL3 (90m), SRTMGL1 (30m)
      - AW3D30 (ALOS 30m)
      - COP30 (Copernicus 30m, 推荐)
      - NASADEM (30m)
    """
    try:
        from bmi_topography import Topography
    except ImportError:
        print("[安装] pip install bmi-topography")
        sys.exit(1)

    if bbox is None:
        bbox = LANZHONG_BBOX

    params = Topography.DEFAULT.copy()
    params.update({"south": bbox["south"], "north": bbox["north"],
                   "west": bbox["west"], "east": bbox["east"],
                   "dem_type": dem_type, "output_format": "GTiff",
                   "api_key": api_key})
    topo = Topography(**params)
    print(f"[下载] {topo.url}")
    path = Path(topo.fetch())
    print(f"[完成] {path} ({path.stat().st_size / 1024 / 1024:.1f} MB)")

    # 复制到 data 目录
    dest = DATA_DIR / f"langzhong_{dem_type.lower()}.tif"
    import shutil
    shutil.copy2(path, dest)
    print(f"[复制] {dest}")
    return dest


def download_gscloud() -> None:
    """地理空间数据云（中国的 GDEM 30m）需手动下载。

    步骤：
      1. 打开 https://www.gscloud.cn/
      2. 注册 → 搜索 "GDEM V3" 或 "ASTER GDEM"
      3. 框选阆中区域 (105.85E-106.00E, 31.50N-31.65N)
      4. 下载 GeoTIFF
      5. 放到 D:\Xunlong\data\ 并重命名
    """
    print("=" * 60)
    print("  地理空间数据云 - 手动下载指引")
    print("=" * 60)
    print()
    print("  1. 打开 https://www.gscloud.cn/")
    print("  2. 注册/登录")
    print("  3. 搜索 'GDEM V3 30m' 或 'SRTM 90m'")
    print("  4. 框选阆中区域:")
    print("     经度: 105.85 - 106.00°E")
    print("     纬度:  31.50 -  31.65°N")
    print("  5. 下载 GeoTIFF")
    print(f"  6. 放到 {DATA_DIR}")
    print()


def use_manual(dem_path: str) -> Path:
    p = Path(dem_path)
    if not p.exists():
        print(f"[错误] 文件不存在: {p}")
        sys.exit(1)
    print(f"[使用] {p}")
    return p


def main():
    p = argparse.ArgumentParser(description="寻龙点穴 - 真实 DEM 下载")
    p.add_argument("mode", choices=["opentopo", "gscloud", "manual"],
                   help="opentopo=自动下载, gscloud=网页指引, manual=手动路径")
    p.add_argument("--api-key", default="demoapikeyot2022", help="OpenTopography API Key")
    p.add_argument("--dem-type", default="COP30",
                   choices=["COP30", "COP90", "SRTMGL1", "SRTMGL3", "AW3D30", "NASADEM"])
    p.add_argument("--dem-path", default=None, help="manual 模式的 DEM 路径")
    args = p.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "opentopo":
        download_bmi_topography(args.api_key, args.dem_type)
    elif args.mode == "gscloud":
        download_gscloud()
    elif args.mode == "manual":
        if not args.dem_path:
            print("[错误] manual 模式需要 --dem-path")
            sys.exit(1)
        use_manual(args.dem_path)

    print()
    print(f"  运行分析: engine\\.venv\\Scripts\\python.exe"
          f" -m engine.cli --dem data\\langzhong_{args.dem_type.lower()}"
          f".tif --water data\\langzhong_rivers.geojson --dragon-vein --top-k 5")


if __name__ == "__main__":
    main()
