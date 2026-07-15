"""Single-candidate multi-factor scoring."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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

from engine.core.scoring.weights import DEFAULT_WEIGHTS, LUNTOU_ONLY_WEIGHTS

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


