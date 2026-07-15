"""综合评分：把各模块输出融合为 0-100 总分。

参考:
  - shanshui-mingtang-fengshui-gis fengshui_score.py
  - 调研报告 99_summary/01_速查卡.md
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


def _find_nearest_shan_by_deg(facing_deg: float) -> tuple[str, float]:
    """给定方位角 → 最近山（中心方位）。"""
    from engine.core.compass_directions import SHAN_TABLE
    f = facing_deg % 360.0
    best, best_diff = None, 360.0
    for name, (c, _syn) in SHAN_TABLE.items():
        d = abs(((f - c + 180.0) % 360.0) - 180.0)
        if d < best_diff:
            best_diff = d
            best = name
    return best, SHAN_TABLE[best][0]


def _facing_to_sit(facing_shan: str) -> str:
    """向 → 坐（取对宫最近山名）。"""
    from engine.core.compass_directions import SHAN_TABLE
    f = SHAN_TABLE[facing_shan][0]
    dual = (f + 180.0) % 360.0
    best, best_diff = None, 360.0
    for name, (c, _syn) in SHAN_TABLE.items():
        d = abs(((dual - c + 180.0) % 360.0) - 180.0)
        if d < best_diff:
            best_diff = d
            best = name
    return best

import numpy as np

from engine.core.acupoint import (
    AcupointCandidate,
    score_stability,
    score_form,
)
from engine.core.compass_directions import (
    classify_facing as _classify_facing,
    facing_cross_check as _facing_cross_check,
    score_compass_purity as _score_compass,
)
from engine.core.four_beasts import FourBeastsScore, score_four_beasts
from engine.core.halo_soil import score_halo_soil as _score_halo_soil
from engine.core.sand_water import SandScore, score_sand_mountain, score_water_relation
from engine.core.star_body import (
    classify_star_body as _classify_star_body,
    classify_xue_star as _classify_xue_star,
    score_xue_star_bonus as _score_xue_star_bonus,
)
from engine.core.terrain_analysis import (
    TerrainMetrics,
    analyze_terrain,
    compute_slope_aspect,
)
from engine.core.water_mouth import (
    best_mouth_for_acupoint,
    find_water_mouths,
    score_mouth_locking,
    score_water_mouth_for_candidate,
)
from engine.core.yuan_yun_xuankong import (
    fly_chart as _fly_xuan_chart,
    score_yun as _score_yun,
)
from engine.core.phenology import (
    PhenologyInputs as _PhenologyInputs,
    score_acupoint_phenology as _score_phenology,
)
from engine.io.dem import DEM
from engine.io.rivers import WaterNetwork
from engine.utils.helpers import clamp_score


# 与评分场 compute_score_grid 对齐（数理 §6 + 候选多因子扩展）
# 四象/形/稳/得水 相对比例保持一致；砂/明堂为候选专有项
DEFAULT_WEIGHTS = {
    "four_beasts": 0.20,  # 峦头主维
    "form": 0.12,
    "sand": 0.10,
    "water": 0.11,       # 得水（中距有情；贴岸另有 bank_penalty）
    "openness": 0.20,    # 明堂开阔（堂心优先，压贴岸光环）
    "stability": 0.12,
    # P0 审核：理气/水口可选进排序（默认开启、权重保守）
    "compass": 0.06,      # 向首合规
    "mouth": 0.06,        # 水口关锁
    "star_body": 0.03,    # 父母山星体吉凶（0–100 归一后）
}

# 兼容：关闭理气加权时用此表（纯峦头）
LUNTOU_ONLY_WEIGHTS = {
    "four_beasts": 0.24,
    "form": 0.13,
    "sand": 0.11,
    "water": 0.12,
    "openness": 0.20,
    "stability": 0.20,
}


@dataclass
class FusedScore:
    """单个候选穴的综合评分结果。"""

    candidate_id: str
    x: float
    y: float
    elevation: float
    form_type: str
    overall: int
    rank: int
    scores: dict[str, int]
    geography: dict[str, Any]
    messages: dict[str, str]
    meta: dict[str, Any] = field(default_factory=dict)


def _score_yaoxia_for_candidate(
    x: float,
    y: float,
    yaoxia_points: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    """过峡入首有情：以候选穴到最近蜂腰鹤膝距离与收窄比给加减分。

    规则（形局，进 overall 固定加减，不进 DEFAULT_WEIGHTS）:
      - 无过峡：0
      - 穴压峡（<40 m）：-2（不宜压在腰上）
      - 中距有情（40–1000 m）：+2~+6（收窄越明显越高）
      - 较远（1000–2500 m）：+1
      - 过远：0
    """
    meta: dict[str, Any] = {
        "yaoxia_evaluated": True,
        "yaoxia_count": len(yaoxia_points),
        "nearest_yaoxia_m": None,
        "nearest_yaoxia_narrow_ratio": None,
        "yaoxia_notes": "无过峡",
    }
    if not yaoxia_points:
        return 0, meta

    best = min(
        yaoxia_points,
        key=lambda p: (float(p.get("x", 0.0)) - x) ** 2
        + (float(p.get("y", 0.0)) - y) ** 2,
    )
    dx = float(best.get("x", 0.0)) - x
    dy = float(best.get("y", 0.0)) - y
    dist_m = float(np.hypot(dx, dy))
    ratio = float(best.get("narrow_ratio", 0.55))
    meta["nearest_yaoxia_m"] = round(dist_m, 1)
    meta["nearest_yaoxia_narrow_ratio"] = round(ratio, 3)

    if dist_m < 40.0:
        bonus = -2
        notes = "穴近压过峡，不宜"
    elif dist_m <= 1000.0:
        if ratio <= 0.40:
            bonus = 6
        elif ratio <= 0.55:
            bonus = 4
        else:
            bonus = 2
        notes = "过峡入首有情"
    elif dist_m <= 2500.0:
        bonus = 1
        notes = "过峡较远，弱关联"
    else:
        bonus = 0
        notes = "过峡过远，无关"
    meta["yaoxia_notes"] = notes
    return int(bonus), meta


def _bearing_from_to(x0: float, y0: float, x1: float, y1: float) -> float:
    """北=0、东=90 的方位角：从 (x0,y0) 指向 (x1,y1)。"""
    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    return float((np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0)


def score_candidate(
    dem: DEM,
    candidate: AcupointCandidate,
    terrain: TerrainMetrics,
    water: WaterNetwork | None = None,
    weights: dict[str, float] | None = None,
    slope_arr: np.ndarray | None = None,
    aspect_arr: np.ndarray | None = None,
    yaoxia_points: list[dict[str, Any]] | None = None,
    long_az_deg: float | None = None,
) -> FusedScore:
    """对单个候选穴进行综合评分。

    Args:
        dem: 完整 DEM
        candidate: 候选穴（行列 + 经纬度）
        terrain: 区域级地形统计
        water: 水系数据
        weights: 自定义权重（默认使用 DEFAULT_WEIGHTS）
        slope_arr / aspect_arr: 预计算坡度/坡向
        yaoxia_points: 可选过峡点列表（来自 dragon_vein.find_yaoxia /
            DragonVeinResult.yaoxia）；None 表示未评估
        long_az_deg: 可选来龙方位（北=0）；由入首点→穴方位注入；None=跳过形理交叉
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS.copy()
    if slope_arr is None or aspect_arr is None:
        slope_arr, aspect_arr = compute_slope_aspect(dem)

    # 1. 提取候选穴周围子 DEM（按 800m x 800m）
    pad_m = 500
    xres, yres = dem.resolution
    from engine.core.terrain_analysis import _is_geographic
    if _is_geographic(dem.crs):
        m_per_unit = 111000.0
    else:
        m_per_unit = 1.0
    pad_px = int(pad_m / (min(xres, yres) * m_per_unit))
    h, w = dem.data.shape
    r0 = max(0, candidate.row - pad_px)
    r1 = min(h, candidate.row + pad_px + 1)
    c0 = max(0, candidate.col - pad_px)
    c1 = min(w, candidate.col + pad_px + 1)
    sub = dem.data[r0:r1, c0:c1]
    from rasterio.windows import Window
    sub_transform = dem.transform * dem.transform.translation(c0, r0)
    sub_dem = DEM(
        data=sub,
        transform=sub_transform,
        crs=dem.crs,
        nodata=dem.nodata,
        # bounds 继承父 DEM 的全局 bounds（仅参考，不影响 pad）
        bounds=dem.bounds,
        resolution=dem.resolution,
    )

    # 2. 四象评分
    four = score_four_beasts(
        sub_dem,
        slope_arr=slope_arr[r0:r1, c0:c1],
        aspect_arr=aspect_arr[r0:r1, c0:c1],
        search_radius_m=300,
    )
    facing_for_open = float((four.details or {}).get("facing", 180.0))

    # 3. 形态评分
    form_score = candidate.form_score

    # 4. 砂山评分（扩大搜索窗，父母/案山常在 500–1500m）
    sand = score_sand_mountain(
        sub_dem,
        slope_arr=slope_arr[r0:r1, c0:c1],
        search_radius_m=800,
    )

    # 5. 水系：得水加分通道（水煞在融合后乘性惩罚）
    water_score = score_water_relation(candidate.x, candidate.y, water, dem)
    water_get = float(water_score.get_score) if not water_score.is_placeholder else 0.0

    # 6. 明堂开阔度（用真实朝向，禁止写死 180）
    from engine.core.acupoint import score_openness

    openness = score_openness(
        sub_dem,
        slope_arr[r0:r1, c0:c1],
        forward_sector_deg=facing_for_open,
        radius_m=700,  # 外明堂略扩，利于河湾堂心
        elev_arr=sub_dem.data,
    )
    # 明堂有情固定加成（堂心开阔优先；贴岸另见 bank_penalty）
    mingtang_bonus = 0
    if openness >= 92:
        mingtang_bonus = 12
    elif openness >= 85:
        mingtang_bonus = 9
    elif openness >= 78:
        mingtang_bonus = 5
    elif openness >= 70:
        mingtang_bonus = 2

    # 7. 稳定性
    stability = score_stability(candidate.local_slope)

    # 7.5 水口识别（P0-4）：量化龙水交媾与水口紧锁度
    mouth_score_val = 0
    mouth_lock_ratio = 0.0
    jiaogou_bonus = 0
    mouth_evaluated = False
    if water is not None and not getattr(water, "empty", True):
        try:
            # 用 DEM 真实高程分天门/地户，避免水系线方向颠倒
            def _mouth_elev_fn(mx: float, my: float) -> float:
                try:
                    return float(dem.sample(mx, my))
                except Exception:
                    return float("nan")

            mouths = find_water_mouths(water, elev_fn=_mouth_elev_fn)
            mouth, _m_dist = best_mouth_for_acupoint(
                water, candidate.x, candidate.y, mouths,
                consideration_radius_m=5000.0,
            )
            if mouth is not None:
                # 真 sand_dist_fn：用 dem.transform 严格反查 + dem.resolution 解析
                # 仅接受 dem.crs 与 water.gdf 一致的情况；不一致返回 -1，由调用方降级处理
                from math import radians, cos, sin

                # 优先看 dem 与 water 是否同 CRS
                _dem_crs = str(getattr(dem, "crs", "")).upper()
                _water_crs = (
                    str(getattr(water, "gdf", None).crs).upper()
                    if getattr(water, "gdf", None) is not None and water.gdf.crs is not None
                    else "EPSG:3857"
                )
                _compatible = (
                    ("3857" in _dem_crs and "3857" in _water_crs)
                    or ("4326" in _dem_crs and "4326" in _water_crs)
                    or (_dem_crs == _water_crs)
                )
                # 从 dem 解析（投影 CRS 假定米制；地理 CRS 假定度 → 用 111000）
                x_res_dem = abs(dem.resolution[0]) if dem.resolution else 30.0
                y_res_dem = abs(dem.resolution[1]) if dem.resolution else 30.0
                if "4326" in _dem_crs:
                    # lat/lon resolution → meters
                    _dem_mpp_x = x_res_dem * 111000.0
                    _dem_mpp_y = y_res_dem * 111000.0
                else:
                    _dem_mpp_x = x_res_dem
                    _dem_mpp_y = y_res_dem

                # dem.transform: rasterio Affine。a = x_res, e = -y_res（since y 轴反向）
                _tf = dem.transform
                # 用 dem.xy → dem.rowcol 严格互转
                cand_xy = (candidate.x, candidate.y)

                def sand_dist_fn(x: float, y: float, bearing_deg: float) -> float:
                    """沿 bearing 方向，最远 1500 m 内最近的"高于母点 8m 以上的砂山"距离。

                    实现：用 dem.rowcol 严格反查 + 沿方位右手法则采样；
                    受 dem.transform 与 dem.resolution 控制，不硬编码 30m；
                    失败边界：返回 1500.0（"过远"），由调用方按 LOCK_BAD 处理。
                    """
                    if not _compatible:
                        return 1500.0
                    # 反查起点像元
                    try:
                        r0, c0 = dem.rowcol(x, y)
                    except Exception:
                        return 1500.0
                    if not (0 <= r0 < dem.data.shape[0] and 0 <= c0 < dem.data.shape[1]):
                        return 1500.0
                    try:
                        cand_elev = float(dem.data[int(r0), int(c0)])
                    except Exception:
                        return 1500.0
                    rad = radians(bearing_deg)
                    # 北=0°，东=+x，西=-x。在图像坐标 (col,row)，北 = -row 方向。
                    # bearing 转图像 dx,dy: dx_step = sin(bearing), dy_step = -cos(bearing)
                    dx_img = sin(rad)            # 列方向(东)
                    dy_img = -cos(rad)           # 行方向(北，即 row 减)
                    # 步长 1 像素 ~ 米 = sqrt(dem_x² + dem_y²)
                    px_step_m = 0.5 * (_dem_mpp_x + _dem_mpp_y)
                    best_d = 1500.0
                    for n in range(1, 60):    # ≤ 60 步 ≈ 1500m（30m 步）/ 较少米
                        d_m = n * px_step_m
                        if d_m > 1500.0:
                            break
                        rr = int(round(r0 + dy_img * n))
                        cc = int(round(c0 + dx_img * n))
                        if not (0 <= rr < dem.data.shape[0] and 0 <= cc < dem.data.shape[1]):
                            continue
                        try:
                            e = float(dem.data[rr, cc])
                        except Exception:
                            continue
                        if e > cand_elev + 8.0:
                            return float(d_m)
                    return best_d

                mouth_lock_ratio = score_mouth_locking(water, mouth, sand_dist_fn)
                mouth_score_val, mouth_msg = score_water_mouth_for_candidate(
                    mouth, lock_ratio=mouth_lock_ratio,
                )
                if mouth.is_jiaogou:
                    jiaogou_bonus = 12  # 龙水交媾点大吉加成
                mouth_msg_text = mouth_msg
                mouth_evaluated = True
            else:
                mouth_msg_text = "无相关水口"
                mouth_evaluated = True
        except Exception:
            mouth_msg_text = "水口评估异常"
    else:
        mouth_msg_text = "无水系数据"

    # 7.6 二十四山向（P1-2）—— 当前朝向是否出卦/兼向
    facing_val = facing_for_open
    compass_score, compass_face = _score_compass(facing_val, base_score=85.0)

    # 7.6.1 形局 × 理气交叉校验（B14）
    # 优先参数 long_az_deg；其次 candidate.long_az_deg（兼容旧注入）
    _long_az = long_az_deg
    if _long_az is None and hasattr(candidate, "long_az_deg"):
        _long_az = getattr(candidate, "long_az_deg", None)
    cross_check_ok: bool | None = None
    cross_check_msg = "未注入 dragon vein 信息"
    cross_check_penalty = 0
    if _long_az is not None and np.isfinite(float(_long_az)):
        cross_check_ok, cross_check_msg = _facing_cross_check(
            facing_val, float(_long_az),
        )
        if cross_check_ok is False:
            cross_check_penalty = -10

    # 7.7 玄空九运（P2）：下卦 + 兼向替卦；城门/零正仍缺 → 不标 implemented
    from datetime import datetime, timezone
    default_year = datetime.now(timezone.utc).year
    xk_chart = None
    xk_score = None
    try:
        from engine.core.yuan_yun_xuankong import year_to_yun, fly_chart_strict, score_yun
        info = year_to_yun(default_year)
        facing_shan, _ = _find_nearest_shan_by_deg(facing_val)
        sit_shan = _facing_to_sit(facing_shan)
        sit_deg = float((facing_val + 180.0) % 360.0)
        xk_chart = fly_chart_strict(
            info.yun, sit_shan, facing_shan,
            shan_deg=sit_deg, facing_deg=float(facing_val),
        )
        xk_score = int(score_yun(info.yun, xk_chart))
    except Exception:
        try:
            from engine.core.yuan_yun_xuankong import year_to_yun
            info = year_to_yun(default_year)
            facing_shan, _ = _find_nearest_shan_by_deg(facing_val)
            sit_shan = _facing_to_sit(facing_shan)
            xk_chart = _fly_xuan_chart(info.yun, sit_shan, facing_shan)
            xk_score = None
        except Exception:
            xk_chart = None
            xk_score = None

    # 7.8 微地形物候（P2-2）—— 代理指标（DEM-only）
    phen = _score_phenology(dem, candidate.row, candidate.col,
                            inputs=_PhenologyInputs(),
                            search_radius_m=30.0)

    # 7.9 晕土识别（P1-5）—— 局部缓变带 + 平缓 TWI 判定
    halo = _score_halo_soil(dem, candidate.row, candidate.col, search_radius_m=30.0)
    halo_score_val = int(round(halo.score)) if halo.score is not None else None

    # 7.10 父母山五行星体（P0 审核）：取**玄武扇区峰**，非子 DEM 全局最高
    star_type = "不清"
    star_eligible = None
    star_score_bonus = 0
    star_body_score = 50  # 0–100，供加权
    try:
        sit_deg = float((facing_for_open + 180.0) % 360.0)
        sh, sw = sub_dem.data.shape
        # 子 DEM 内相对中心
        cr_sub = int(candidate.row - r0)
        cc_sub = int(candidate.col - c0)
        cr_sub = int(np.clip(cr_sub, 0, sh - 1))
        cc_sub = int(np.clip(cc_sub, 0, sw - 1))
        cand_elev_local = float(dem.data[candidate.row, candidate.col])
        mpx_s = abs(sub_dem.resolution[0])
        mpy_s = abs(sub_dem.resolution[1])
        from engine.core.terrain_analysis import _is_geographic
        if _is_geographic(sub_dem.crs):
            mpx_s *= 111000.0
            mpy_s *= 111000.0
        best_star = None
        best_elev = -1e18
        # 玄武扇区：坐向 ±50°，距 50–500 m，取最高点
        for rr in range(sh):
            for cc in range(sw):
                z = float(sub_dem.data[rr, cc])
                if not np.isfinite(z) or z < cand_elev_local + 4.0:
                    continue
                dx = (cc - cc_sub) * mpx_s
                dy = (cr_sub - rr) * mpy_s  # 北正
                dist = float(np.hypot(dx, dy))
                if dist < 50.0 or dist > 500.0:
                    continue
                brg = float((np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0)
                d_ang = abs(((brg - sit_deg + 180.0) % 360.0) - 180.0)
                if d_ang > 50.0:
                    continue
                if z > best_elev:
                    best_elev = z
                    best_star = (rr, cc)
        if best_star is not None:
            star = _classify_star_body(
                sub_dem, int(best_star[0]), int(best_star[1]), search_radius_m=300.0,
            )
            star_type = star.type
            star_eligible = star.is_xuanwu_eligible
            if star.type in ("火星", "廉贞"):
                star_score_bonus = -8
                star_body_score = 25
            elif star.type in ("金星", "木星", "水星"):
                star_score_bonus = +5
                star_body_score = 78
            else:
                star_body_score = 50
        else:
            star_type = "无父母山"
            star_body_score = 45
    except Exception:
        star_type = "未评估"
        star_eligible = None
        star_body_score = 50

    # 7.10b 穴星本体（P2）：穴位本身五行星体
    xue_star_type = "未评估"
    xue_star_bonus = 0
    xue_star_notes = ""
    try:
        xue = _classify_xue_star(
            dem, candidate.row, candidate.col,
            search_radius_m=80.0,
            form_hint=candidate.form_type,
        )
        xue_star_type = xue.type
        xue_star_bonus, xue_star_notes = _score_xue_star_bonus(xue)
    except Exception:
        xue_star_type = "未评估"
        xue_star_bonus = 0
        xue_star_notes = "穴星评估失败"

    # 7.11 过峡入首（A1-余）—— 固定加减分进 overall
    if yaoxia_points is not None:
        yaoxia_bonus, yaoxia_meta = _score_yaoxia_for_candidate(
            candidate.x, candidate.y, yaoxia_points,
        )
    else:
        yaoxia_bonus = 0
        yaoxia_meta = {
            "yaoxia_evaluated": False,
            "yaoxia_count": 0,
            "nearest_yaoxia_m": None,
            "nearest_yaoxia_narrow_ratio": None,
            "yaoxia_notes": "未注入龙脉过峡信息",
        }

    # 8. 加权融合（数理 §6 + 审核 P0：理气/水口可选加权）
    from engine.core.water_model import fuse_field_with_sha

    # 峦头六维 + compass/mouth/star_body（DEFAULT_WEIGHTS 已含；可传 LUNTOU_ONLY_WEIGHTS 关闭）
    raw = {
        "four_beasts": four.combined,
        "form": form_score,
        "sand": sand.score,
        "water": water_get,  # 仅 S_得水
        "openness": openness,
        "stability": stability,
        "water_sha": float(water_score.sha_penalty),
        "mouth": float(mouth_score_val),
        "compass": float(compass_score),
        "star_body": float(star_body_score),
    }
    if water_score.is_placeholder:
        w_local = {k: v for k, v in weights.items() if k != "water"}
    else:
        w_local = dict(weights)
    # 无水口/未评估时不占权重
    if not mouth_evaluated or mouth_score_val <= 0:
        w_local.pop("mouth", None)
    if star_type in ("未评估", "无父母山", "不清"):
        # 仍保留极小权重或剔除
        w_local.pop("star_body", None)
    # 只累加 raw 中有的键
    w_local = {k: v for k, v in w_local.items() if k in raw and k != "water_sha"}
    w_sum = sum(w_local.values()) or 1.0
    overall = sum(float(raw[k]) * w_local[k] for k in w_local) / w_sum
    if not water_score.is_placeholder:
        overall = fuse_field_with_sha(overall, float(water_score.sha_penalty))
    # 龙水交媾固定加成
    overall += jiaogou_bonus
    # 父母山五行星体奖惩（额外固定档，与 star_body 权重互补）
    overall += star_score_bonus
    # 穴星本体固定加减
    overall += xue_star_bonus
    # 平坦明堂有情加成（对标参考图河湾橙心）
    overall += mingtang_bonus
    # 贴岸惩罚 + 堂心中距水加成：穴宜离岸入堂，忌最高分挤河岸
    bank_penalty = 0
    hall_water_bonus = 0
    d_w = water_score.nearest_distance_m
    if (
        not water_score.is_placeholder
        and d_w is not None
        and np.isfinite(float(d_w))
    ):
        dw = float(d_w)
        if dw < 90:
            bank_penalty = 14
        elif dw < 140:
            bank_penalty = 10
        elif dw < 200:
            bank_penalty = 6
        elif dw < 260:
            bank_penalty = 3
        # 堂心有情：开阔 + 中距界水
        if openness >= 80 and 220.0 <= dw <= 800.0:
            hall_water_bonus = 6
        elif openness >= 75 and 180.0 <= dw <= 900.0:
            hall_water_bonus = 4
        elif openness >= 70 and 200.0 <= dw <= 1000.0:
            hall_water_bonus = 2
    overall += hall_water_bonus - bank_penalty
    # 晕土加成（最高 +8，正相关）
    if halo_score_val is not None:
        overall += (halo_score_val - 50) * 0.16  # 50-100 分缩放 → 0-8 分
    # 过峡入首加减
    overall += yaoxia_bonus
    # B14 形理交叉：反局骑龙罚分
    overall += cross_check_penalty

    return FusedScore(
        candidate_id="",
        x=candidate.x,
        y=candidate.y,
        elevation=candidate.elevation,
        form_type=candidate.form_type,
        overall=clamp_score(overall),
        rank=0,
        scores={k: int(v) for k, v in raw.items() if k != "water_sha"}
        | {
            "water": int(water_get),
            "water_get": int(water_score.get_score),
            "water_sha": int(water_score.sha_penalty),
            "mouth": int(mouth_score_val),
            "compass": int(compass_score),
            "xuankong": int(xk_score) if xk_score is not None else None,
            # 【P-3 集成】晕土与五行星体加减分已计入 overall
            "halo_soil": halo_score_val,
            "star_body_bonus": int(star_score_bonus),
            "star_body": int(star_body_score),
            "xue_star_bonus": int(xue_star_bonus),
            "mingtang_bonus": int(mingtang_bonus),
            "bank_penalty": int(bank_penalty),
            "hall_water_bonus": int(hall_water_bonus),
            # 【A1-余】朝抱已并入 sand；过峡加减计入 overall
            "embrace_left": (
                int(round(sand.embrace_left))
                if getattr(sand, "embrace_left", None) is not None
                else None
            ),
            "embrace_right": (
                int(round(sand.embrace_right))
                if getattr(sand, "embrace_right", None) is not None
                else None
            ),
            "yaoxia_bonus": int(yaoxia_bonus),
            "cross_check_penalty": int(cross_check_penalty),
        },
        meta={
            "weighted_dims": list((weights or {"four_beasts": 0.28}).keys()),
            "xuankong_implemented": False,
            "phenology_is_proxy": True,
            "mouth_evaluated": mouth_evaluated,
            "star_body_type": star_type,
            "star_body_is_xuanwu_eligible": star_eligible,
            "xue_star_type": xue_star_type,
            "xue_star_notes": xue_star_notes,
            "qinglong_morph": (four.details or {}).get("qinglong_morph"),
            "baihu_tame": (four.details or {}).get("baihu_tame"),
            "zhuque_viewshed": (four.details or {}).get("zhuque_viewshed"),
            "halo_soil_notes": halo.notes if halo else None,
            "cross_check_ok": cross_check_ok,
            "cross_check_msg": cross_check_msg,
            "long_az_deg": (
                round(float(_long_az), 2)
                if _long_az is not None and np.isfinite(float(_long_az))
                else None
            ),
            "embrace_in_sand": True,
            "yaoxia_evaluated": yaoxia_meta.get("yaoxia_evaluated", False),
            "yaoxia_count": yaoxia_meta.get("yaoxia_count", 0),
            "yaoxia_notes": yaoxia_meta.get("yaoxia_notes"),
        },
        geography={
            "tpi": round(candidate.tpi, 2),
            "twi": round(candidate.twi, 2),
            "local_slope": round(candidate.local_slope, 2),
            "back_mountain_height_m": round(sand.back_mountain_height, 1) if not np.isnan(sand.back_mountain_height) else None,
            "back_mountain_distance_m": round(sand.back_mountain_distance_m, 1) if not np.isnan(sand.back_mountain_distance_m) else None,
            "nearest_water_m": round(water_score.nearest_distance_m, 1) if water_score.nearest_distance_m is not None else None,
            "nearest_water_dir": water_score.direction,
            "water_get": int(water_score.get_score),
            "water_sha": int(water_score.sha_penalty),
            "water_form": water_score.form or {},
            "qinglong": four.qinglong,
            "baihu": four.baihu,
            "zhuque": four.zhuque,
            "xuanwu": four.xuanwu,
            "compass_shan": compass_face.shan,
            "compass_dev_deg": round(compass_face.deviation_deg, 2),
            "compass_jian_xiang": compass_face.is_jian_xiang,
            "compass_chu_gua": compass_face.is_chu_gua,
            "water_mouth_lock_ratio": round(mouth_lock_ratio, 3),
            "xuankong_yun": int(xk_chart.yun) if xk_chart is not None else None,
            "xuankong_yuan": xk_chart.yuan if xk_chart is not None else None,
            "xuankong_shan_gua": xk_chart.shan_gua if xk_chart is not None else None,
            "xuankong_facing_gua": xk_chart.facing_gua if xk_chart is not None else None,
            # P2：下卦/替卦已排星；城门/零正仍缺 → implemented 仍 False
            "xuankong_shan_star_at_facing": (
                int(xk_chart.shan_star_at_facing)
                if xk_chart is not None and xk_chart.shan_star_at_facing is not None
                else None
            ),
            "xuankong_facing_star_at_facing": (
                int(xk_chart.facing_star_at_facing)
                if xk_chart is not None and xk_chart.facing_star_at_facing is not None
                else None
            ),
            "xuankong_notes": xk_chart.notes if xk_chart is not None else None,
            "xuankong_implemented": False,
            "phenology_total": phen.get("total"),
            "phenology_ndvi": phen.get("ndvi_score"),
            "phenology_moisture": phen.get("moisture_score"),
            "phenology_is_proxy": True,  # 标记为 DEM 代理值
            # 【修复 D1】mouth 是否评估，False 时表示未参加计算
            "mouth_evaluated": mouth_evaluated,
            # 【修复 D1】权重信息：明示 mouth/compass/xuankong 仅展示，不进 overall
            "weighted_dims": list((weights or {"four_beasts": 0.28}).keys()),
            # 【A1-余】朝抱 / 过峡
            "embrace_left": sand.embrace_left,
            "embrace_right": sand.embrace_right,
            "nearest_yaoxia_m": yaoxia_meta.get("nearest_yaoxia_m"),
            "nearest_yaoxia_narrow_ratio": yaoxia_meta.get(
                "nearest_yaoxia_narrow_ratio"
            ),
            "yaoxia_count": yaoxia_meta.get("yaoxia_count", 0),
            "yaoxia_evaluated": yaoxia_meta.get("yaoxia_evaluated", False),
            "long_az_deg": (
                round(float(_long_az), 2)
                if _long_az is not None and np.isfinite(float(_long_az))
                else None
            ),
            "cross_check_ok": cross_check_ok,
        },
        messages={
            "sand": sand.message,
            "water": water_score.message,
            "compass": compass_face.notes,
            "mouth": mouth_msg_text,
            "xuankong": xk_chart.notes if xk_chart else "",
            "phenology": phen.get("notes", ""),
            "yaoxia": yaoxia_meta.get("yaoxia_notes", ""),
            "cross_check": cross_check_msg,
        },
    )


def find_and_rank_candidates(
    dem: DEM,
    water: WaterNetwork | None = None,
    top_k: int = 10,
    min_score: int = 50,
    weights: dict[str, float] | None = None,
    *,
    dragon_vein=None,
    primary_dragon=None,
    return_context: bool = False,
) -> list[FusedScore] | tuple[list[FusedScore], dict[str, Any]]:
    """先定龙再点穴：主来龙入首邻域优先，综合分混入龙对齐。

    Args:
        dem / water / top_k / min_score / weights: 同前
        dragon_vein: 可选已算好的 DragonVeinResult（避免重复 ~2min 全量龙脉）
        primary_dragon: 可选 PrimaryDragon
        return_context: True 时额外返回 {dragon_vein, primary_dragon, qi_grid}
    """
    from engine.core.acupoint import search_candidates
    from engine.core.dragon_vein import (
        analyze_dragon_vein,
        select_primary_dragon,
        dragon_alignment_score,
        reorient_primary_to_hole,
        _m_per_px_dem,
    )

    # E.5：全 NaN / 无有效像元 → 空结果，不崩溃、不造高分
    if dem is None or dem.data is None or not np.isfinite(dem.data).any():
        empty_ctx = {"dragon_vein": None, "primary_dragon": None, "qi_grid": None}
        return ([], empty_ctx) if return_context else []

    slope_arr, aspect_arr = compute_slope_aspect(dem)
    try:
        terrain = analyze_terrain(dem)
    except ValueError:
        empty_ctx = {"dragon_vein": None, "primary_dragon": None, "qi_grid": None}
        return ([], empty_ctx) if return_context else []

    from engine.core.four_beasts_detect import compute_score_grid, find_score_peak
    from engine.core.four_beasts_detect import WATER_BAN_BUFFER_M

    _ban_m = float(WATER_BAN_BUFFER_M)
    mpx, mpy = _m_per_px_dem(dem)

    # —— Step 1: 生气场（理论最优锚 = 热峰）——
    try:
        qi_grid = compute_score_grid(dem, water=water)
    except Exception:
        qi_grid = None

    peak_rc: tuple[int, int, float] | None = None
    if qi_grid is not None:
        peak = find_score_peak(qi_grid)
        if peak is not None:
            peak_rc = (int(peak[0]), int(peak[1]), float(peak[2]))

    # —— Step 2: 定龙（相对热峰选主脊；全量 D8 可接受数分钟）——
    yaoxia_points: list[dict[str, Any]] = []
    dv = dragon_vein
    primary = primary_dragon
    try:
        if dv is None:
            dv = analyze_dragon_vein(dem, min_length_m=120.0, water=water)
        yaoxia_points = list(getattr(dv, "yaoxia", None) or [])
        ar = peak_rc[0] if peak_rc else dem.data.shape[0] // 2
        ac = peak_rc[1] if peak_rc else dem.data.shape[1] // 2
        # Tier 2 G：热峰 + 入首双信号锚
        from engine.core.dragon_vein import dual_signal_anchor
        ent = getattr(dv, "entrance_point", None) if dv is not None else None
        peak_pt = (peak_rc[0], peak_rc[1]) if peak_rc else None
        anchor = dual_signal_anchor(peak_pt, ent, mpx, mpy)
        if anchor is not None:
            ar, ac = int(anchor[0]), int(anchor[1])
        primary = select_primary_dragon(
            dem, water=water, dragon_vein=dv,
            anchor_row=ar, anchor_col=ac,
        )
    except Exception:
        if primary is None:
            primary = primary_dragon
        yaoxia_points = list(yaoxia_points)

    entrance_xy: tuple[float, float] | None = None
    flow_az: float | None = None
    if primary is not None:
        entrance_xy = primary.entrance_xy
        flow_az = float(primary.flow_azimuth_deg)
    elif dv is not None and getattr(dv, "entrance_xy", None) is not None:
        entrance_xy = (float(dv.entrance_xy[0]), float(dv.entrance_xy[1]))

    def _primary_for_cand(c: AcupointCandidate):
        """相对候选重定向主龙。"""
        if primary is None:
            return None
        try:
            return reorient_primary_to_hole(
                dem, primary, int(c.row), int(c.col), water=water,
            )
        except Exception:
            return primary

    # —— Step 3: 搜穴 ——
    # P0：TWI 参与搜穴——用龙脉汇流累积（若有）
    flow_acc_arr = None
    if dv is not None and getattr(dv, "flow_acc", None) is not None:
        flow_acc_arr = dv.flow_acc

    # 明堂/热力区：步长更密 + NMS 略松，避免河湾橙心无点
    cands = search_candidates(
        dem, flow_acc=flow_acc_arr, tpi_radius_m=100, tpi_threshold=0.0,
        max_candidates=160, step=3, water=water, ban_buffer_m=_ban_m,
        qi_grid=qi_grid, qi_min_percentile=40.0,
        min_dist_m=140.0,
    )
    if len(cands) < 5:
        cands = search_candidates(
            dem, flow_acc=flow_acc_arr, tpi_radius_m=80, tpi_threshold=0.0,
            max_candidates=180, step=2, water=water, ban_buffer_m=_ban_m,
            qi_grid=qi_grid, qi_min_percentile=25.0,
            min_dist_m=120.0,
        )
    if len(cands) < 3:
        cands = search_candidates(
            dem, flow_acc=flow_acc_arr, tpi_radius_m=60, tpi_threshold=0.0,
            max_candidates=180, step=2, water=water, ban_buffer_m=_ban_m,
            qi_grid=qi_grid, qi_min_percentile=10.0,
            min_dist_m=100.0,
        )

    def _long_az_for(c: AcupointCandidate) -> float | None:
        # 来龙方位 = 相对本穴定向后的 源→入首（气向）
        p = _primary_for_cand(c)
        if p is not None:
            return float(p.flow_azimuth_deg)
        if flow_az is not None:
            return flow_az
        if entrance_xy is None:
            return None
        return _bearing_from_to(entrance_xy[0], entrance_xy[1], c.x, c.y)

    def _is_on_water(c: AcupointCandidate) -> bool:
        if water is None or getattr(water, "empty", True):
            return False
        try:
            if water.intersects(c.x, c.y, buffer_m=_ban_m):
                return True
        except Exception:
            pass
        return False

    def _qi_at(c: AcupointCandidate) -> float:
        if qi_grid is None:
            return 50.0
        r, col = int(c.row), int(c.col)
        if 0 <= r < qi_grid.shape[0] and 0 <= col < qi_grid.shape[1]:
            v = float(qi_grid[r, col])
            if np.isfinite(v):
                return v
        return 0.0

    # 与 search_candidates 一致：TPI 阈值随分辨率缩放
    from engine.core.terrain_analysis import _is_geographic as _is_geo_cell
    _xr, _yr = abs(dem.resolution[0]), abs(dem.resolution[1])
    if _is_geo_cell(dem.crs):
        _cell_m = float(min(_xr, _yr) * 111000.0)
    else:
        _cell_m = float(min(_xr, _yr))

    # 与 search_candidates 一致：有 flow_acc 则算 TWI 栅格供强制候选使用
    _twi_arr = None
    if flow_acc_arr is not None and np.any(flow_acc_arr > 0):
        try:
            from engine.core.acupoint import compute_twi
            _twi_arr = compute_twi(dem, flow_acc_arr)
        except Exception:
            _twi_arr = None

    # 水禁栅格：热峰若落在禁带，吸附到邻近干地高 qi
    _water_ban_grid = None
    if water is not None and not getattr(water, "empty", True):
        try:
            from engine.core.four_beasts_detect import water_distance_rasters
            _d_ban, _water_ban_grid = water_distance_rasters(
                dem, water, ban_buffer_m=float(_ban_m),
            )
        except Exception:
            _water_ban_grid = None

    def _snap_dry_high_qi(
        pr: int, pc: int, *, max_m: float = 280.0,
    ) -> tuple[int, int, float]:
        """保证落在干地；优先邻域 qi 最大。返回 (row, col, qi)。"""
        h0, w0 = dem.data.shape
        pr = int(np.clip(pr, 0, h0 - 1))
        pc = int(np.clip(pc, 0, w0 - 1))
        q0 = 0.0
        if qi_grid is not None and np.isfinite(qi_grid[pr, pc]):
            q0 = float(qi_grid[pr, pc])
        on_ban = (
            _water_ban_grid is not None
            and _water_ban_grid.shape == dem.data.shape
            and bool(_water_ban_grid[pr, pc])
        )
        if not on_ban and np.isfinite(dem.data[pr, pc]):
            return pr, pc, q0
        rad = max(2, int(round(max_m / max(min(mpx, mpy), 1.0))))
        best = None
        best_q = -1e18
        for rr in range(max(0, pr - rad), min(h0, pr + rad + 1)):
            for cc in range(max(0, pc - rad), min(w0, pc + rad + 1)):
                if not np.isfinite(dem.data[rr, cc]):
                    continue
                if (
                    _water_ban_grid is not None
                    and _water_ban_grid.shape == dem.data.shape
                    and _water_ban_grid[rr, cc]
                ):
                    continue
                d_m = float(np.hypot((rr - pr) * mpy, (cc - pc) * mpx))
                if d_m > max_m:
                    continue
                q = 0.0
                if qi_grid is not None and np.isfinite(qi_grid[rr, cc]):
                    q = float(qi_grid[rr, cc])
                # 近 + 高 qi
                sc = q - d_m * 0.02
                if sc > best_q:
                    best_q = sc
                    best = (rr, cc, q)
        if best is None:
            return pr, pc, q0
        return int(best[0]), int(best[1]), float(best[2])

    _edge_margin_px = max(4, int(round(120.0 / max(min(mpx, mpy), 1.0))))

    def _is_edge_cell(pr: int, pc: int) -> bool:
        h0, w0 = dem.data.shape
        return (
            pr < _edge_margin_px
            or pc < _edge_margin_px
            or pr >= h0 - _edge_margin_px
            or pc >= w0 - _edge_margin_px
        )

    def _make_cand_at(pr: int, pc: int, form_boost: float = 0.0) -> AcupointCandidate | None:
        from engine.core.acupoint import (
            classify_form, score_form, AcupointCandidate as _AC,
        )
        if not (0 <= pr < dem.data.shape[0] and 0 <= pc < dem.data.shape[1]):
            return None
        if not np.isfinite(dem.data[pr, pc]):
            return None
        # 图缘假点（常堆到 UI 左上角）禁止作为候选
        if _is_edge_cell(pr, pc):
            return None
        tpi_p = float(
            __import__("engine.core.terrain_analysis", fromlist=["tpi"]).tpi(
                dem, radius_m=100
            )[pr, pc]
        )
        if not np.isfinite(tpi_p):
            tpi_p = 0.0
        ls = float(slope_arr[pr, pc]) if np.isfinite(slope_arr[pr, pc]) else 2.0
        ft = classify_form(tpi_p, ls, cell_size_m=_cell_m)
        form_sc = int(score_form(tpi_p, ft, cell_size_m=_cell_m))
        twi_v = 0.0
        if _twi_arr is not None and np.isfinite(_twi_arr[pr, pc]):
            twi_v = float(_twi_arr[pr, pc])
            # 与 search_candidates 相同的 TWI 微调
            if 2.0 <= twi_v <= 10.0:
                form_sc = int(min(100, form_sc + 6))
            elif twi_v > 14.0:
                form_sc = int(max(0, form_sc - 8))
            elif 0 < twi_v < 1.0:
                form_sc = int(max(0, form_sc - 3))
        # 高 form_boost（热峰）时再抬：保证进排序前列
        form_sc = int(round(max(form_sc, form_boost)))
        if form_boost >= 70:
            form_sc = int(min(100, max(form_sc, 88)))
        px, py = dem.xy(pr, pc)
        if not (np.isfinite(px) and np.isfinite(py)):
            return None
        ac = _AC(
            row=pr, col=pc, x=float(px), y=float(py),
            elevation=float(dem.data[pr, pc]),
            tpi=tpi_p, twi=float(twi_v),
            form_type=ft,
            form_score=form_sc,
            local_slope=ls,
        )
        if _is_on_water(ac):
            return None
        return ac

    def _add_cand(c: AcupointCandidate | None, *, min_sep_m: float = 90.0) -> bool:
        if c is None:
            return False
        for i, e in enumerate(cands):
            d = float(np.hypot((e.row - c.row) * mpy, (e.col - c.col) * mpx))
            if d < min_sep_m:
                # 已有邻近点：更高 form 则替换
                if c.form_score > e.form_score:
                    cands[i] = c
                    return True
                return False
        cands.insert(0, c)
        return True

    # 入首邻域强制注入（先龙后穴）；拒绝图缘入首（假龙源/尾常贴边）
    if primary is not None:
        er, ec = primary.entrance_row, primary.entrance_col
        if not _is_edge_cell(er, ec):
            er, ec, _ = _snap_dry_high_qi(er, ec)
            if not _is_edge_cell(er, ec):
                ent_cand = _make_cand_at(er, ec, form_boost=60.0)
                _add_cand(ent_cand, min_sep_m=80.0)
            if qi_grid is not None:
                rad_px = max(3, int(round(600.0 / max(min(mpx, mpy), 1.0))))
                r0 = max(0, er - rad_px)
                r1 = min(qi_grid.shape[0], er + rad_px + 1)
                c0 = max(0, ec - rad_px)
                c1 = min(qi_grid.shape[1], ec + rad_px + 1)
                sub = qi_grid[r0:r1, c0:c1]
                if sub.size and np.isfinite(sub).any():
                    filled = np.where(np.isfinite(sub), sub, -np.inf)
                    # 窗内取 qi 最大且非图缘
                    best_loc = None
                    best_q = -1e18
                    for lr in range(sub.shape[0]):
                        for lc in range(sub.shape[1]):
                            pr, pc = int(r0 + lr), int(c0 + lc)
                            if _is_edge_cell(pr, pc):
                                continue
                            q = float(filled[lr, lc])
                            if q > best_q:
                                best_q = q
                                best_loc = (pr, pc, q)
                    if best_loc is not None:
                        pr, pc, qv = _snap_dry_high_qi(best_loc[0], best_loc[1])
                        if not _is_edge_cell(pr, pc):
                            loc = _make_cand_at(pr, pc, form_boost=max(float(qv), 70.0))
                            _add_cand(loc, min_sep_m=80.0)

    # 热峰 + 明堂高 qi 多种子强制注入（橙心必须有备选）
    peak_cand = None
    if peak_rc is not None:
        pr, pc, psc = peak_rc
        pr, pc, qv = _snap_dry_high_qi(pr, pc, max_m=320.0)
        boost = max(float(psc), float(qv), 85.0)
        peak_cand = _make_cand_at(pr, pc, form_boost=boost)
        if peak_cand is not None:
            # 更新 peak_rc 为吸附后坐标（供 is_qi_peak 匹配）
            peak_rc = (pr, pc, boost)
            _add_cand(peak_cand, min_sep_m=60.0)
            peak_cand = next(
                (c for c in cands if c.row == pr and c.col == pc), peak_cand
            )

    # 明堂高 qi 区「铺点」：网格取局部最高，强制空间分散
    # （禁止全部挤在热峰 200 m 内，右边大片橙心也要有候选）
    if qi_grid is not None and np.isfinite(qi_grid).any():
        try:
            valid = np.isfinite(qi_grid)
            if _water_ban_grid is not None and _water_ban_grid.shape == qi_grid.shape:
                valid = valid & (~_water_ban_grid)
            # 图缘不铺点
            em = _edge_margin_px
            valid[:em, :] = False
            valid[-em:, :] = False
            valid[:, :em] = False
            valid[:, -em:] = False
            q_valid = qi_grid[valid]
            if q_valid.size > 30:
                # 略降分位：覆盖更广的橙色明堂腹地
                q_thr = float(np.nanpercentile(q_valid, 58))
                hot = valid & (qi_grid >= q_thr)
                # 网格边长约 350–450 m：每格最多 1 个最高 qi 干点
                cell_m = 380.0
                cell_px = max(4, int(round(cell_m / max(min(mpx, mpy), 1.0))))
                h0, w0 = qi_grid.shape
                seeds: list[tuple[float, int, int]] = []
                for r0 in range(0, h0, cell_px):
                    for c0 in range(0, w0, cell_px):
                        r1 = min(h0, r0 + cell_px)
                        c1 = min(w0, c0 + cell_px)
                        block = hot[r0:r1, c0:c1]
                        if not block.any():
                            continue
                        sub = np.where(block, qi_grid[r0:r1, c0:c1], -np.inf)
                        li = int(np.argmax(sub))
                        lr, lc = np.unravel_index(li, sub.shape)
                        rr, cc = int(r0 + lr), int(c0 + lc)
                        qv = float(qi_grid[rr, cc])
                        seeds.append((qv, rr, cc))
                # 按 qi 降序；间距 ≥ 280 m 注入（拉开、覆盖橙心）
                seeds.sort(key=lambda t: -t[0])
                n_seed = 0
                peak_r = peak_rc[0] if peak_rc else None
                peak_c = peak_rc[1] if peak_rc else None
                near_peak_n = 0  # 热峰 350 m 内最多 2 个（含峰本身）
                for qv, rr, cc in seeds:
                    if n_seed >= 14:
                        break
                    rr, cc, q2 = _snap_dry_high_qi(rr, cc, max_m=120.0)
                    if _is_edge_cell(rr, cc):
                        continue
                    if peak_r is not None:
                        d_peak = float(np.hypot(
                            (rr - peak_r) * mpy, (cc - peak_c) * mpx,
                        ))
                        if d_peak < 350.0:
                            if near_peak_n >= 2:
                                continue
                            near_peak_n += 1
                    ac = _make_cand_at(rr, cc, form_boost=max(float(q2), float(qv), 72.0))
                    # 更大间距：避免 6/9/C-001 叠在峰上
                    if _add_cand(ac, min_sep_m=280.0):
                        n_seed += 1
        except Exception:
            pass

    # qi 分位：高 qi 保底龙分
    qi_p85, qi_p95 = 70.0, 85.0
    if qi_grid is not None and np.isfinite(qi_grid).any():
        valid_q = qi_grid[np.isfinite(qi_grid)]
        if valid_q.size > 20:
            qi_p85 = float(np.nanpercentile(valid_q, 85))
            qi_p95 = float(np.nanpercentile(valid_q, 95))

    def _fuse_overall(form_sc: float, qv: float, d_align: float) -> int:
        # 理论：橙心（明堂 qi）≈最优 → qi 权更大；龙只作贴脊加分
        # 高 qi 时抬高龙分下限
        da = d_align
        if qv >= qi_p95:
            da = max(da, 82.0)
        elif qv >= qi_p85:
            da = max(da, 70.0)
        elif qv >= 60.0:
            da = max(da, 55.0)
        return clamp_score(0.30 * form_sc + 0.56 * qv + 0.14 * da)

    # 剔除无效坐标 / 图缘点（防止 UI 堆左上角）
    cands = [
        c for c in cands
        if c is not None
        and np.isfinite(c.x) and np.isfinite(c.y)
        and not _is_edge_cell(int(c.row), int(c.col))
    ]

    results: list[FusedScore] = []
    peak_fused: FusedScore | None = None
    for i, c in enumerate(cands):
        if _is_on_water(c):
            continue
        if not (np.isfinite(c.x) and np.isfinite(c.y)):
            continue
        fused = score_candidate(
            dem, c, terrain, water, weights, slope_arr, aspect_arr,
            yaoxia_points=yaoxia_points,
            long_az_deg=_long_az_for(c),
        )
        fused.candidate_id = f"C-{i+1:03d}"
        if water is not None and not getattr(water, "empty", True):
            try:
                if water.intersects(c.x, c.y, buffer_m=0):
                    continue
            except Exception:
                pass
        qv = _qi_at(c)
        d_align = 50.0
        d_meta: dict[str, float] = {}
        p_loc = _primary_for_cand(c)
        if p_loc is not None:
            d_meta = dragon_alignment_score(
                int(c.row), int(c.col), p_loc, mpx, mpy,
            )
            d_align = float(d_meta.get("dragon_align", 50.0))
        fused.overall = _fuse_overall(float(fused.overall), qv, d_align)
        if fused.meta is not None:
            fused.meta["qi_field"] = round(qv, 1)
            fused.meta["dragon_align"] = round(d_align, 1)
            if d_meta:
                fused.meta["dist_entrance_m"] = round(
                    float(d_meta.get("dist_entrance_m", 0)), 1
                )
                fused.meta["dist_ridge_m"] = round(
                    float(d_meta.get("dist_ridge_m", 0)), 1
                )
            p_info = p_loc or primary
            if p_info is not None:
                fused.meta["primary_dragon"] = {
                    "ridge_idx": int(p_info.ridge_idx),
                    "flow_az": round(float(p_info.flow_azimuth_deg), 1),
                    "sit": round(float(p_info.sit_deg), 1),
                    "facing": round(float(p_info.facing_deg), 1),
                    "source_rc": [int(p_info.source_row), int(p_info.source_col)],
                    "entrance_rc": [int(p_info.entrance_row), int(p_info.entrance_col)],
                    "method": getattr(p_info, "method", ""),
                }
                fused.meta["long_az_deg"] = round(float(p_info.flow_azimuth_deg), 1)
            if peak_cand is not None and c.row == peak_cand.row and c.col == peak_cand.col:
                fused.meta["is_qi_peak"] = True
                peak_fused = fused
        if fused.overall >= min_score:
            results.append(fused)

    if not results and cands:
        scored: list[FusedScore] = []
        for i, c in enumerate(cands):
            if _is_on_water(c):
                continue
            fused = score_candidate(
                dem, c, terrain, water, weights, slope_arr, aspect_arr,
                yaoxia_points=yaoxia_points,
                long_az_deg=_long_az_for(c),
            )
            fused.candidate_id = f"C-{i+1:03d}"
            qv = _qi_at(c)
            d_align = 50.0
            p_loc = _primary_for_cand(c)
            if p_loc is not None:
                d_meta = dragon_alignment_score(
                    int(c.row), int(c.col), p_loc, mpx, mpy,
                )
                d_align = float(d_meta.get("dragon_align", 50.0))
            fused.overall = _fuse_overall(float(fused.overall), qv, d_align)
            scored.append(fused)
        scored.sort(key=lambda x: -x.overall)
        results = scored[: max(top_k, 5)]

    results.sort(key=lambda x: -x.overall)

    # 热峰强制进入结果池
    if peak_fused is not None:
        in_list = any(
            (getattr(r, "meta") or {}).get("is_qi_peak")
            or (abs(r.x - peak_fused.x) < 1e-6 and abs(r.y - peak_fused.y) < 1e-6)
            for r in results
        )
        if not in_list:
            results = [r for r in results if not (
                abs(r.x - peak_fused.x) < 1e-6 and abs(r.y - peak_fused.y) < 1e-6
            )]
            if peak_fused.overall < 78:
                peak_fused.overall = clamp_score(max(float(peak_fused.overall), 80))
            results.insert(0, peak_fused)
            results.sort(key=lambda x: -x.overall)

    # 空间分散 top_k：分数优先，但相邻候选 ≥ ~320 m
    # 避免 6/9/C-001 全挤在场评最高点，明堂右侧空无一穴
    from engine.core.terrain_analysis import _is_geographic as _is_geo_final
    _geo = _is_geo_final(dem.crs)

    def _sep_m(a: FusedScore, b: FusedScore) -> float:
        if _geo:
            mid_lat = (a.y + b.y) / 2.0
            dx = (a.x - b.x) * 111_000.0 * max(0.2, abs(np.cos(np.radians(mid_lat))))
            dy = (a.y - b.y) * 111_000.0
            return float(np.hypot(dx, dy))
        return float(np.hypot(a.x - b.x, a.y - b.y))

    min_sep_final = 320.0
    diverse: list[FusedScore] = []
    # 热峰优先入选
    if peak_fused is not None:
        diverse.append(peak_fused)
    for r in results:
        if peak_fused is not None and (
            (getattr(r, "meta") or {}).get("is_qi_peak")
            or (abs(r.x - peak_fused.x) < 1e-6 and abs(r.y - peak_fused.y) < 1e-6)
        ):
            continue
        if any(_sep_m(r, k) < min_sep_final for k in diverse):
            continue
        diverse.append(r)
        if len(diverse) >= top_k:
            break
    # 若分散后不足 top_k，放宽间距再补
    if len(diverse) < top_k:
        for r in results:
            if any(
                abs(r.x - k.x) < 1e-9 and abs(r.y - k.y) < 1e-9 for k in diverse
            ):
                continue
            if any(_sep_m(r, k) < min_sep_final * 0.55 for k in diverse):
                continue
            diverse.append(r)
            if len(diverse) >= top_k:
                break
    results = diverse

    for i, r in enumerate(results):
        r.rank = i + 1
        r.candidate_id = f"C-{i+1:03d}"
    out = results[:top_k]
    ctx = {
        "dragon_vein": dv,
        "primary_dragon": primary,
        "qi_grid": qi_grid,
        "qi_peak_rowcol": (peak_rc[0], peak_rc[1]) if peak_rc else None,
    }
    if return_context:
        return out, ctx
    return out


def to_geojson(results: list[FusedScore]) -> dict[str, Any]:
    """输出候选穴为 GeoJSON FeatureCollection。"""
    features = []
    for r in results:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r.x, r.y]},
            "properties": {
                "id": r.candidate_id,
                "rank": r.rank,
                "overall_score": r.overall,
                "elevation_m": r.elevation,
                "form_type": r.form_type,
                "scores": r.scores,
                "geography": r.geography,
                "messages": r.messages,
            },
        })
    return _sanitize({"type": "FeatureCollection", "features": features})


def to_json(results: list[FusedScore], metadata: dict | None = None) -> dict[str, Any]:
    """输出 JSON 报告。"""
    out = {
        "metadata": metadata or {},
        "candidates": [
            {
                "id": r.candidate_id,
                "rank": r.rank,
                "x": r.x,
                "y": r.y,
                "elevation_m": r.elevation,
                "form_type": r.form_type,
                "overall_score": r.overall,
                "scores": r.scores,
                "geography": r.geography,
                "messages": r.messages,
                "meta": r.meta or {},
            }
            for r in results
        ],
    }
    return _sanitize(out)


def _sanitize(obj):
    """递归清洗 JSON：inf/NaN→None，numpy 标量/数组→原生类型，dict 键也转原生。

    修复：FastAPI jsonable_encoder 遇 numpy.int32 键/值会 500
    （'numpy.int32' object is not iterable）。
    """
    import math

    def _key(k):
        if isinstance(k, (np.integer,)):
            return int(k)
        if isinstance(k, (np.floating,)):
            v = float(k)
            if math.isnan(v) or math.isinf(v):
                return str(k)
            # JSON 对象键最终须为 str；先转 Python 数再由 encoder 处理
            return int(v) if v == int(v) else v
        if isinstance(k, (bytes, bytearray)):
            return k.decode("utf-8", errors="replace")
        if isinstance(k, np.bool_):
            return bool(k)
        return k

    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, dict):
        return {_key(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (int,)):
        # 排除 bool（bool 是 int 子类，上面已处理）
        return int(obj)
    # set / 其它可迭代但非 str
    if isinstance(obj, set):
        return [_sanitize(v) for v in obj]
    return obj
