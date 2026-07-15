"""Four-beasts / shaozu gate for candidate ranking."""
from __future__ import annotations

from typing import Any

import numpy as np

from engine.io.dem import DEM

def _gate_beasts_for_hole(
    dem: DEM,
    row: int,
    col: int,
    water=None,
    *,
    primary_dragon=None,
    dragon_vein=None,
    require_shaozu_higher: bool = True,
    min_side_beasts: int = 2,
) -> tuple[bool, str, dict[str, Any]]:
    """四象/祖山门禁：识别失败则候选淘汰。

    成功经验（如 C-007）：少祖高于玄武 → 来龙顺气剥换而下，可结穴。
    硬条件：
      1. 少祖、玄武均可定位
      2. 青龙/白虎/朱雀至少 min_side_beasts 个可定位（四象不残）
      3. 少祖 elev ≥ 玄武 elev − 容差（顺气；可关）
    """
    from engine.core.four_beasts_detect import detect_four_beasts

    info: dict[str, Any] = {"beasts_ok": False}
    try:
        fb = detect_four_beasts(
            dem,
            center_row=int(row),
            center_col=int(col),
            water=water,
            primary_dragon=primary_dragon,
            dragon_vein=dragon_vein,
            use_incoming_vein=True,
        )
    except Exception as e:
        info["reason"] = f"detect_error:{e}"
        return False, "detect_error", info

    beasts = (fb.meta or {}).get("beasts") or {}
    sz = beasts.get("shaozu")
    xw = beasts.get("xuanwu")
    zq = beasts.get("zhuque")
    ql = beasts.get("qinglong")
    bh = beasts.get("baihu")
    info["beasts_present"] = {
        "shaozu": sz is not None,
        "xuanwu": xw is not None,
        "zhuque": zq is not None,
        "qinglong": ql is not None,
        "baihu": bh is not None,
    }
    if sz is not None:
        info["shaozu_elev_m"] = sz.get("elev_m")
        info["shaozu_dist_m"] = sz.get("dist_m")
    if xw is not None:
        info["xuanwu_elev_m"] = xw.get("elev_m")
        info["xuanwu_dist_m"] = xw.get("dist_m")

    if xw is None:
        info["reason"] = "no_xuanwu"
        return False, "no_xuanwu", info
    if sz is None:
        info["reason"] = "no_shaozu"
        return False, "no_shaozu", info

    side_n = sum(1 for p in (zq, ql, bh) if p is not None)
    if side_n < int(min_side_beasts):
        info["reason"] = f"incomplete_four_beasts({side_n}<{min_side_beasts})"
        info["side_n"] = side_n
        return False, "incomplete_four_beasts", info

    elev_bonus = 0
    try:
        e_sz = float(sz.get("elev_m"))
        e_xw = float(xw.get("elev_m"))
        if np.isfinite(e_sz) and np.isfinite(e_xw):
            dh = e_sz - e_xw
            info["shaozu_minus_xuanwu_m"] = round(dh, 1)
            # 容差 2m：同高可过；明显更低 = 逆剥/假祖
            if require_shaozu_higher and dh < -2.0:
                info["reason"] = "shaozu_not_higher_than_xuanwu"
                return False, "shaozu_not_higher", info
            # 顺气加分：少祖明显高于玄武
            if dh >= 5.0:
                elev_bonus = int(min(10, 3 + dh / 8.0))
            elif dh >= 0.0:
                elev_bonus = 2
    except Exception:
        pass

    info["beasts_ok"] = True
    info["reason"] = "ok"
    info["shaozu_higher_bonus"] = elev_bonus
    info["facing"] = float(getattr(fb, "facing", 0.0) or 0.0)
    info["sit"] = float(getattr(fb, "sit", 0.0) or 0.0)
    return True, "ok", info

