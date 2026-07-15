"""Demo: 端到端调用传统风水模块输出可读报告。

输出每个候选穴的：
  - 形态（窝/钳/乳/突）+ 综合分
  - 四兽 + 穴 + 朝 + 山
  - 二十四山向（出卦/兼向）
  - 玄空九运盘（山星 / 向星）
  - 水口与龙水交媾
  - 微地形物候（N/A 或 NDVI/soil moisture）
"""
from __future__ import annotations

import argparse
import json
import sys

from engine.io.dem import load_dem
from engine.io.rivers import load_water
from engine.core.fengshui_score import find_and_rank_candidates


def render_candidate(idx: int, c: dict, meta: dict = None) -> str:
    """格式化单个候选穴为可读报告。"""
    geo = c.get('geography', {})
    scores = c.get('scores', {})
    msgs = c.get('messages', {})

    lines: list[str] = []
    lines.append("")
    lines.append(f"========== 候选穴 #{idx} {c.get('id')} (rank={c.get('rank')}) ==========")
    lines.append(f"坐标: ({c.get('x'):.1f}, {c.get('y'):.1f})  海拔: {c.get('elevation_m'):.1f} m")
    lines.append(f"形态: {c.get('form_type')}    综合分: {c.get('overall_score')}")
    lines.append("")

    lines.append("【峦头】")
    ql = geo.get('qinglong'); bh = geo.get('baihu'); zq = geo.get('zhuque'); xw = geo.get('xuanwu')
    lines.append(f"  四象: 青龙={ql}  白虎={bh}  朱雀={zq}  玄武={xw}")
    lines.append(f"  父母山: 高 {geo.get('back_mountain_height_m')} m, 距 {geo.get('back_mountain_distance_m')} m")
    lines.append(f"  局部: TPI={geo.get('tpi')}, TWI={geo.get('twi')}, 坡度={geo.get('local_slope')}°")
    lines.append(f"  砂山总评: {scores.get('sand')}  明堂开阔度: {scores.get('openness')}")
    if msgs.get('sand'):
        lines.append(f"  砂说: {msgs['sand']}")
    lines.append("")

    lines.append("【理气】")
    lines.append(f"  二十四山: {geo.get('compass_shan')}（偏差 {geo.get('compass_dev_deg')}°）")
    lines.append(f"    出卦={geo.get('compass_chu_gua')}, 兼向={geo.get('compass_jian_xiang')}")
    if msgs.get('compass'):
        lines.append(f"    {msgs['compass']}")
    lines.append(f"  玄空: {geo.get('xuankong_yuan')} {geo.get('xuankong_yun')}运")
    lines.append(f"    山星到向: {geo.get('xuankong_shan_star_at_facing')}, 向星到向: {geo.get('xuankong_facing_star_at_facing')}")
    if msgs.get('xuankong'):
        lines.append(f"    {msgs['xuankong']}")
    lines.append("")

    lines.append("【水法】")
    lines.append(f"  最近水: {geo.get('nearest_water_dir')}侧 {geo.get('nearest_water_m')} m")
    lines.append(f"  得水: {scores.get('water_get')}, 水煞: {scores.get('water_sha')}")
    if geo.get('water_form'):
        wf = geo['water_form']
        non_zero = {k: round(v, 2) for k, v in wf.items() if isinstance(v, (int, float)) and v != 0}
        if non_zero:
            lines.append(f"  水形态: {non_zero}")
    if geo.get('water_mouth_lock_ratio') is not None:
        lines.append(f"  水口锁紧度: {geo['water_mouth_lock_ratio']}")
        if msgs.get('mouth'):
            lines.append(f"    {msgs['mouth']}")
    if msgs.get('water'):
        lines.append(f"  水说: {msgs['water']}")
    lines.append("")

    lines.append("【穴位・微地形】")
    lines.append(f"  物候综合: {geo.get('phenology_total')}  (NDVI={geo.get('phenology_ndvi')}, 湿度={geo.get('phenology_moisture')})")
    if msgs.get('phenology'):
        lines.append(f"    {msgs['phenology']}")

    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description="端到端风水 API demo")
    parser.add_argument('--dem', default=r'D:\Xunlong\data\langzhong_cop30.tif')
    parser.add_argument('--water', default=r'D:\Xunlong\data\langzhong_rivers_osm.geojson')
    parser.add_argument('--top-k', type=int, default=5)
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    dem = load_dem(args.dem)
    water = load_water(args.water) if args.water else None
    results = find_and_rank_candidates(dem, water, top_k=args.top_k, min_score=0)
    obj = {"candidates": [
        {
            "id": r.candidate_id, "rank": r.rank,
            "x": r.x, "y": r.y, "elevation_m": r.elevation,
            "form_type": r.form_type, "overall_score": r.overall,
            "scores": r.scores, "geography": r.geography, "messages": r.messages,
        }
        for r in results
    ]}
    print("=" * 60)
    print(f"寻龙点穴·终评报告    ({args.dem})")
    print(f"候选数: {len(obj['candidates'])}")
    print("=" * 60)
    for i, c in enumerate(obj["candidates"], 1):
        print(render_candidate(i, c))


if __name__ == "__main__":
    main()
