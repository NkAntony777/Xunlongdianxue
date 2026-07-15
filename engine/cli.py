"""CLI 入口：通过命令行使用寻龙引擎。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from engine.io.dem import load_dem
from engine.io.rivers import load_water
from engine.core.fengshui_score import find_and_rank_candidates, to_json, to_geojson
from engine.core.dragon_vein import analyze_dragon_vein
from engine.core.terrain_analysis import analyze_terrain


def main():
    p = argparse.ArgumentParser(
        prog="xunlong",
        description="寻龙点穴地形分析引擎 - CLI",
    )
    p.add_argument("--dem", required=True, help="DEM GeoTIFF 路径")
    p.add_argument("--water", help="水系 GeoJSON 路径（可选）")
    p.add_argument("--top-k", type=int, default=10, help="返回候选数")
    p.add_argument("--min-score", type=int, default=40, help="最低综合分阈值")
    p.add_argument("--out-json", help="JSON 报告输出路径")
    p.add_argument("--out-geojson", help="GeoJSON 候选穴输出路径")
    p.add_argument("--dragon-vein", action="store_true", help="同时进行龙脉识别")
    p.add_argument("--quiet", action="store_true", help="不打印中间信息")
    args = p.parse_args()

    if not args.quiet:
        print(f"[1/4] 加载 DEM: {args.dem}")
    dem = load_dem(args.dem)
    if not args.quiet:
        print(f"      网格: {dem.shape[1]} x {dem.shape[0]} 像素, "
              f"分辨率 {dem.resolution[0]:.1f} m, 范围 {dem.bounds}")

    water = None
    if args.water:
        if not args.quiet:
            print(f"[2/4] 加载水系: {args.water}")
        water = load_water(args.water)
        if not args.quiet:
            print(f"      要素数: {len(water.gdf)}")

    if not args.quiet:
        print(f"[3/4] 计算地形统计...")
    metrics = analyze_terrain(dem)
    if not args.quiet:
        print(f"      均高 {metrics.mean_elevation:.0f} m, "
              f"高差 {metrics.relief:.0f} m, "
              f"主坡向 {metrics.dominant_aspect}, "
              f"类型 {metrics.terrain_position}")

    if not args.quiet:
        print(f"[4/4] 搜索候选穴 (top_k={args.top_k}, min_score={args.min_score})...")
    results = find_and_rank_candidates(
        dem, water, top_k=args.top_k, min_score=args.min_score
    )
    if not args.quiet:
        print(f"      找到 {len(results)} 个候选穴")
        for r in results:
            print(
                f"        {r.candidate_id} ({r.x:.4f}, {r.y:.4f}) "
                f"h={r.elevation:.0f}m form={r.form_type} score={r.overall}"
            )

    # 输出
    if args.out_json:
        metadata = {
            "dem": str(args.dem),
            "water": str(args.water) if args.water else None,
            "bbox": list(dem.bounds),
            "resolution_m": dem.resolution[0],
            "terrain": {
                "mean_elevation": metrics.mean_elevation,
                "relief": metrics.relief,
                "mean_slope": metrics.mean_slope,
                "dominant_aspect": metrics.dominant_aspect,
                "terrain_position": metrics.terrain_position,
            },
        }
        report = to_json(results, metadata=metadata)
        Path(args.out_json).write_text(json.dumps(report, ensure_ascii=False, indent=2))
        if not args.quiet:
            print(f"      JSON 报告: {args.out_json}")

    if args.out_geojson:
        gj = to_geojson(results)
        Path(args.out_geojson).write_text(json.dumps(gj, ensure_ascii=False, indent=2))
        if not args.quiet:
            print(f"      GeoJSON:   {args.out_geojson}")

    if args.dragon_vein:
        if not args.quiet:
            print(f"[+] 龙脉识别...")
        dv = analyze_dragon_vein(dem)
        if not args.quiet:
            print(f"      山脊线条数: {len(dv.ridge_lines)}")
            print(f"      入首点: {dv.entrance_xy}")
            for i, r in enumerate(dv.major_ridges[:5], 1):
                print(f"        龙脉 #{i}: 长 {r.length_m:.0f} m, "
                      f"蜿蜒度 {r.sinuosity:.2f}, "
                      f"最高 {r.max_elevation:.0f} m")

    return 0


if __name__ == "__main__":
    sys.exit(main())
