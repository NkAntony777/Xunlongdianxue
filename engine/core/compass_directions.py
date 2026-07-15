"""二十四山向与兼向/出卦判别。

传统理据：
  - 罗盘「二十四山」：每山 15°，含三元龙（天、地、人）。
    壬子癸 丑艮寅 甲卯乙 辰巽巳 丙午丁 未坤申 庚酉辛 戌乾亥
  - 「立向」以正中线为「正向」，偏 0-4.5° 仍算「本山向」；
    偏 4.5-9° 为「兼向」；超 9°（即越过邻山中心一半以上）为「出卦」大凶。
  - 「净阴净阳」：天元龙（子午卯酉） + 人元龙（寅申巳亥） + 地元龙（辰戌丑未）
    「净阴」「净阳」「阴阳驳杂」—— 与兼向辨真。

坐标约定（本项目）：
  - 北 = 0°, 东 = 90°, 南 = 180°, 西 = 270°，顺时针递增。

二十四山中心方位（北起子，按子/壬/癸/丑/艮/寅... 排列，每 15° 一山，
但每卦中部为「三元龙」）：
  子(0°)、癸(15°)、丑(30°)、艮(45°)、寅(60°)、甲(75°)、
  卯(90°)、乙(105°)、辰(120°)、巽(135°)、巳(150°)、丙(165°)、
  午(180°)、丁(195°)、未(210°)、坤(225°)、申(240°)、庚(255°)、
  酉(270°)、辛(285°)、戌(300°)、乾(315°)、亥(330°)、壬(345°)

实现：
  - 24 山向表（中心方位 + 三元龙 + 山名）
  - facing → 落山 + 是否出卦 / 兼向 / 本山
  - 「三五颠倒」粗略：来龙与朝向的交叉角
"""
from __future__ import annotations

from dataclasses import dataclass

# 24 山中心方位（北起子，顺时针）
# 三元龙：无常派/沈氏 — 每卦 地元-天元-人元（B8 审核 2026-07 校准）
SHAN_TABLE: dict[str, tuple[float, str]] = {
    # 山名 : (中心方位度, 三元龙)
    "\u5b50": (   0.0, "天"),  # 子  坎天元
    "\u7678": (  15.0, "人"),  # 癸  坎人元  【修】原误「天」
    "\u4e11": (  30.0, "地"),  # 丑  艮地元
    "\u826e": (  45.0, "天"),  # 艮  艮天元  【修】原误「地」
    "\u5bc5": (  60.0, "人"),  # 寅  艮人元
    "\u7532": (  75.0, "地"),  # 甲  震地元  【修】原误「人」
    "\u536f": (  90.0, "天"),  # 卯  震天元
    "\u4e59": ( 105.0, "人"),  # 乙  震人元  【修】原误「天」
    "\u8fb0": ( 120.0, "地"),  # 辰  巽地元
    "\u5dfd": ( 135.0, "天"),  # 巽  巽天元  【修】原误「地」
    "\u5df3": ( 150.0, "人"),  # 巳  巽人元
    "\u4e19": ( 165.0, "地"),  # 丙  离地元  【修】原误「人」
    "\u5348": ( 180.0, "天"),  # 午  离天元
    "\u4e01": ( 195.0, "人"),  # 丁  离人元  【修】原误「天」
    "\u672a": ( 210.0, "地"),  # 未  坤地元
    "\u5764": ( 225.0, "天"),  # 坤  坤天元  【修】原误「地」
    "\u7533": ( 240.0, "人"),  # 申  坤人元
    "\u5e9a": ( 255.0, "地"),  # 庚  兑地元  【修】原误「人」
    "\u9149": ( 270.0, "天"),  # 酉  兑天元
    "\u8f9b": ( 285.0, "人"),  # 辛  兑人元  【修】原误「天」
    "\u620c": ( 300.0, "地"),  # 戌  乾地元
    "\u4e7e": ( 315.0, "天"),  # 乾  乾天元  【修】原误「地」
    "\u4ea5": ( 330.0, "人"),  # 亥  乾人元
    "\u58ec": ( 345.0, "地"),  # 壬  坎地元  【修】原误「人」
}

# 三元龙集合（与 SHAN_TABLE 一致；供飞星同元龙检索）
TIANYUAN = {
    "\u5b50", "\u5348", "\u536f", "\u9149",  # 子午卯酉
    "\u4e7e", "\u5764", "\u826e", "\u5dfd",  # 乾坤艮巽
}
DIYUAN = {
    "\u7532", "\u5e9a", "\u58ec", "\u4e19",  # 甲庚壬丙
    "\u8fb0", "\u620c", "\u4e11", "\u672a",  # 辰戌丑未
}
RENYUAN = {
    "\u5bc5", "\u7533", "\u5df3", "\u4ea5",  # 寅申巳亥
    "\u4e59", "\u8f9b", "\u4e01", "\u7678",  # 乙辛丁癸
}

# 玄空阴阳（与三元龙表一致；净阴净阳 / 飞星起星共用）
YANG_TIAN = {"\u4e7e", "\u5764", "\u826e", "\u5dfd"}  # 乾坤艮巽
YIN_TIAN = {"\u5b50", "\u5348", "\u536f", "\u9149"}   # 子午卯酉
YANG_DI = {"\u7532", "\u5e9a", "\u58ec", "\u4e19"}    # 甲庚壬丙
YIN_DI = {"\u8fb0", "\u620c", "\u4e11", "\u672a"}     # 辰戌丑未
YANG_REN = {"\u5bc5", "\u7533", "\u5df3", "\u4ea5"}   # 寅申巳亥
YIN_REN = {"\u4e59", "\u8f9b", "\u4e01", "\u7678"}    # 乙辛丁癸


# 阈值：偏 0-4.5°=本卦偏，4.5-9°=兼向，>9°=出卦
QIE_LIMIT_DEG = 4.5
CHU_GUA_LIMIT_DEG = 9.0


@dataclass
class MountainFacing:
    """二十四山立向判别结果。"""

    shan: str          # 山名（如"子"）
    target_deg: float  # 山中心方位（度）
    actual_facing_deg: float  # 用户传入朝向（度）
    deviation_deg: float      # 实际偏离（度）
    is_jian_xiang: bool       # 是否兼向（4.5-9°）
    is_chu_gua: bool          # 是否出卦（>9°）
    san_yuan: str             # 三元龙（天/地/人）
    yin_yang_status: str      # 净阴/净阳/驳杂
    notes: str


def find_nearest_shan(facing_deg: float) -> tuple[str, float]:
    """给定方位角 → 最接近的山名 + 该山中心方位。"""
    f = facing_deg % 360.0
    best = None
    best_diff = 360.0
    best_target = 0.0
    for name, (c, _syn) in SHAN_TABLE.items():
        d = abs(((f - c + 180.0) % 360.0) - 180.0)
        if d < best_diff:
            best_diff = d
            best = name
            best_target = c
    return best, float(best_target)


def classify_facing(facing_deg: float) -> MountainFacing:
    """判别是否为本山 / 兼向 / 出卦，及三元龙 + 净阴净阳。

    简化：本算法只针对「立向」（facing 不区分坐向）。
    """
    f = facing_deg % 360.0
    shan, target = find_nearest_shan(f)
    dev = abs(((f - target + 180.0) % 360.0) - 180.0)

    is_jian = QIE_LIMIT_DEG < dev <= CHU_GUA_LIMIT_DEG
    is_chu = dev > CHU_GUA_LIMIT_DEG

    san = SHAN_TABLE[shan][1]

    # 净阴净阳
    if (shan in YIN_TIAN) or (shan in YIN_REN) or (shan in YIN_DI):
        yinyang = "净阴"
    elif (shan in YANG_TIAN) or (shan in YANG_REN) or (shan in YANG_DI):
        yinyang = "净阳"
    else:
        yinyang = "驳杂"

    notes_parts = []
    if is_chu:
        notes_parts.append(f"出卦({dev:.1f}\u00b0)，大凶不宜")
    elif is_jian:
        notes_parts.append(f"兼向({dev:.1f}\u00b0)，需用七十二穿山 / 六十透地")
    else:
        notes_parts.append(f"本山({dev:.1f}\u00b0)")

    return MountainFacing(
        shan=shan,
        target_deg=float(target),
        actual_facing_deg=float(f),
        deviation_deg=float(dev),
        is_jian_xiang=is_jian,
        is_chu_gua=is_chu,
        san_yuan=san,
        yin_yang_status=yinyang,
        notes="; ".join(notes_parts),
    )


def score_compass_purity(facing_deg: float, *, base_score: float = 80.0) -> tuple[float, MountainFacing]:
    """根据二十四山立向的质量打分（0-100）。

    - 出卦 → 大罚
    - 兼向 → 中罚
    - 本山 → 满分
    - 净阴净阳额外微调
    """
    f = classify_facing(facing_deg)
    score = base_score
    if f.is_chu_gua:
        score -= 50
    elif f.is_jian_xiang:
        score -= 15
    if f.yin_yang_status in ("净阴", "净阳"):
        score += 5
    elif f.yin_yang_status == "驳杂":
        score -= 3
    score = float(max(0.0, min(100.0, score)))
    return score, f


def facing_cross_check(
    facing_deg: float,
    long_incoming_az_deg: float | None,
) -> tuple[bool, str]:
    """来龙与立向一致性校验。

    龙从 long_incoming_az 来穴，到穴后立向 facing。理想龙向与坐向（facing+180）
    偏差约 0°（纯骑龙）或 30°-60°（横龙/斜落）。偏向 ≥ 90° 视为反局。

    Returns:
        (pass, reason_str)
    """
    if long_incoming_az_deg is None:
        return True, "无来龙方位信息，跳过"
    sit = (facing_deg + 180.0) % 360.0
    cross = abs(((long_incoming_az_deg - sit + 180.0) % 360.0) - 180.0)
    if cross < 30:
        return True, f"正龙骑龙，{cross:.0f}\u00b0 一致性佳"
    if cross < 60:
        return True, f"斜落骑龙，{cross:.0f}\u00b0 在可接受区"
    if cross < 90:
        return False, f"斜落偏大，{cross:.0f}\u00b0 需复核"
    return False, f"反局：龙与坐向偏差 {cross:.0f}\u00b0 \u226b 90\u00b0，凶"
