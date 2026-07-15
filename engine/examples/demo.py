"""端到端 Demo：合成数据 → 全流程 → 输出报告。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from engine.io.dem import load_dem
from engine.io.rivers import load_water
from engine.core.fengshui_score import find_and_rank_candidates, to_json, to_geojson
from engine.core.dragon_vein import analyze_dragon_vein
from engine.core.terrain_analysis import analyze_terrain


HERE = Path(__file__).parent
FIXTURES = HERE.parent / "tests" / "fixtures"
OUTPUT = HERE / "output"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  寻龙点穴引擎 - 端到端 Demo")
    print("=" * 60)

    # 1. 合成数据（如果还没有）
    dem_path = FIXTURES / "synth_dem.tif"
    rivers_path = FIXTURES / "synth_rivers.geojson"
    if not dem_path.exists() or not rivers_path.exists():
        from engine.tests.fixtures.make_synthetic import make_synthetic_dem, make_synthetic_rivers

        print("\n[生成] 合成数据 ...")
        make_synthetic_dem(dem_path)
        make_synthetic_rivers(rivers_path)
    print(f"  DEM:    {dem_path}")
    print(f"  Rivers: {rivers_path}")

    # 2. 加载
    print("\n[加载] ...")
    dem = load_dem(dem_path)
    water = load_water(rivers_path)
    print(f"  DEM: {dem.shape[1]} x {dem.shape[0]} 像素, 分辨率 {dem.resolution[0]:.1f} m")
    print(f"  高程: {dem.data[np.isfinite(dem.data)].min():.0f} - "
          f"{dem.data[np.isfinite(dem.data)].max():.0f} m")
    print(f"  水系要素: {len(water.gdf)}")

    # 3. 地形分析
    print("\n[地形分析] ...")
    m = analyze_terrain(dem)
    print(f"  均高:    {m.mean_elevation:.0f} m")
    print(f"  高差:    {m.relief:.0f} m")
    print(f"  均坡度:  {m.mean_slope:.1f}°")
    print(f"  主坡向:  {m.dominant_aspect} ({m.aspect_degree:.0f}°)")
    print(f"  位置:    {m.terrain_position}")
    print(f"  粗糙度:  {m.terrain_roughness:.3f}")

    # 4. 候选穴搜索
    print("\n[候选穴搜索] ...")
    results = find_and_rank_candidates(dem, water, top_k=10, min_score=40)
    print(f"  找到 {len(results)} 个候选穴：\n")
    print(f"  {'ID':<6} {'X (m)':>9} {'Y (m)':>9} {'高程':>6} {'形态':<5} "
          f"{'龙':>3} {'虎':>3} {'雀':>3} {'武':>3} {'水':>3} "
          f"{'形':>3} {'砂':>3} {'开':>3} {'稳':>3} {'总':>3}")
    print("  " + "-" * 100)
    for r in results:
        s = r.scores
        print(
            f"  {r.candidate_id:<6} {r.x:>9.0f} {r.y:>9.0f} {r.elevation:>6.0f} "
            f"{r.form_type:<5} "
            f"{s.get('four_beasts', 0):>3} {s.get('form', 0):>3} "
            f"{s.get('water', 0):>3} {s.get('sand', 0):>3} "
            f"{s.get('openness', 0):>3} {s.get('stability', 0):>3} "
            f"{r.overall:>3}"
        )

    # 5. 龙脉识别
    print("\n[龙脉识别] ...")
    dv = analyze_dragon_vein(dem)
    print(f"  提取山脊线: {len(dv.ridge_lines)} 条")
    print(f"  一级主脉:   {len(dv.major_ridges)} 条")
    if dv.entrance_xy:
        ex, ey = dv.entrance_xy
        print(f"  入首点:     ({ex:.0f}, {ey:.0f})")
    print(f"\n  Top 5 主脉：")
    for i, r in enumerate(dv.major_ridges[:5], 1):
        print(f"    #{i}: 长 {r.length_m:.0f} m, "
              f"蜿蜒度 {r.sinuosity:.2f}, "
              f"高 {r.max_elevation:.0f} m, "
              f"显著度 {r.feature_significance:.2e}")

    # 6. 输出
    print("\n[输出] ...")
    metadata = {
        "dem": str(dem_path),
        "water": str(rivers_path),
        "bbox": list(dem.bounds),
        "resolution_m": dem.resolution[0],
        "terrain": {
            "mean_elevation": m.mean_elevation,
            "relief": m.relief,
            "mean_slope": m.mean_slope,
            "dominant_aspect": m.dominant_aspect,
            "terrain_position": m.terrain_position,
        },
    }
    report = to_json(results, metadata=metadata)
    out_json = OUTPUT / "report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"  JSON 报告: {out_json}")

    out_geo = OUTPUT / "candidates.geojson"
    out_geo.write_text(json.dumps(to_geojson(results), ensure_ascii=False, indent=2))
    print(f"  GeoJSON:   {out_geo}")

    print("\n" + "=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
