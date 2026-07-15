"""Four-beasts / shaozu gate for candidate ranking.

Strict path: need 玄武+少祖 (side sand optional).
Soft path (peak / high-qi 明堂): never drop solely for incomplete beasts;
「少祖高于玄武」is bonus/penalty, not a hard kill.
"""
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
    require_shaozu_higher: bool = False,
    min_side_beasts: int = 0,
    soft_keep: bool = False,
) -> tuple[bool, str, dict[str, Any]]:
    """四象/祖山门禁。

    成功经验（顺气）：少祖高于玄武 → 加分。
    - soft_keep=True（热峰/高 qi 堂心）：缺祖/玄/侧砂不淘汰，只记 meta。
    - require_shaozu_higher：默认 False；若 True 则祖低于玄硬否决（兼容旧测）。
    """
    from engine.core.four_beasts_detect import detect_four_beasts

    info: dict[str, Any] = {
        "beasts_ok": False,
        "soft_keep": bool(soft_keep),
    }
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
        if soft_keep:
            info["beasts_ok"] = False
            return True, "soft_pass_detect_error", info
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

    elev_bonus = 0

    if xw is None:
        info["reason"] = "no_xuanwu"
        if soft_keep:
            info["beasts_ok"] = False
            info["facing"] = float(getattr(fb, "facing", 0.0) or 0.0)
            info["sit"] = float(getattr(fb, "sit", 0.0) or 0.0)
            info["shaozu_higher_bonus"] = 0
            return True, "soft_no_xuanwu", info
        return False, "no_xuanwu", info
    if sz is None:
        info["reason"] = "no_shaozu"
        if soft_keep:
            info["beasts_ok"] = False
            info["facing"] = float(getattr(fb, "facing", 0.0) or 0.0)
            info["sit"] = float(getattr(fb, "sit", 0.0) or 0.0)
            info["shaozu_higher_bonus"] = 0
            return True, "soft_no_shaozu", info
        return False, "no_shaozu", info

    side_n = sum(1 for p in (zq, ql, bh) if p is not None)
    info["side_n"] = side_n
    if side_n < int(min_side_beasts) and not soft_keep:
        info["reason"] = f"incomplete_four_beasts({side_n}<{min_side_beasts})"
        return False, "incomplete_four_beasts", info

    try:
        e_sz = float(sz.get("elev_m"))
        e_xw = float(xw.get("elev_m"))
        if np.isfinite(e_sz) and np.isfinite(e_xw):
            dh = e_sz - e_xw
            info["shaozu_minus_xuanwu_m"] = round(dh, 1)
            if require_shaozu_higher and dh < -2.0 and not soft_keep:
                info["reason"] = "shaozu_not_higher_than_xuanwu"
                return False, "shaozu_not_higher", info
            # 软约束：顺气加分 / 逆剥小罚（不淘汰堂心）
            if dh >= 5.0:
                elev_bonus = int(min(10, 3 + dh / 8.0))
            elif dh >= 0.0:
                elev_bonus = 2
            elif dh < -2.0:
                elev_bonus = -4
                info["shaozu_lower_soft"] = True
            else:
                elev_bonus = 0
    except Exception:
        pass

    info["beasts_ok"] = True
    info["reason"] = "ok"
    info["shaozu_higher_bonus"] = elev_bonus
    info["facing"] = float(getattr(fb, "facing", 0.0) or 0.0)
    info["sit"] = float(getattr(fb, "sit", 0.0) or 0.0)
    return True, "ok", info
