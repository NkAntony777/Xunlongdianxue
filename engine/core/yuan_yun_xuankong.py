"""三元九运、玄空挨星（山星 + 向星）、旺山旺向查表。

传统理据（《沈氏玄空学》《飞星派》）：
  - 三元：上元 (运 1-3)、中元 (运 4-6)、下元 (运 7-9)，每元 60 年。
  - 九运：每运 20 年。起算 1864 甲子。九运：2024–2043。
  - 下卦排盘（无常派/沈氏小玄空主流，B8 审核锁定）：
      * 运盘：当运星入中，**一律顺飞**
      * 山/向盘：运盘坐/向宫之星入中；
        顺逆 = 星数→元旦盘宫→与坐/向**同元龙**之山阴阳（阳顺阴逆）
      * 五黄入中：取坐山/向本身阴阳（D2 决策）
      * 替卦/城门/零正神：未实现 → xuankong_implemented=false

研究用；勿当作专业排盘软件。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Yuan(str, Enum):
    """三元。"""
    SHANG = "\u4e0a\u5143"
    ZHONG = "\u4e2d\u5143"
    XIA   = "\u4e0b\u5143"


@dataclass
class YunInfo:
    """单运信息。"""
    yun: int              # 1..9
    yuan: Yuan            # 上/中/下
    year_start: int
    year_end: int


# 9 运定义（西历）
YUN_TABLE: list[YunInfo] = [
    YunInfo(1, Yuan.SHANG, 1864, 1883),
    YunInfo(2, Yuan.SHANG, 1884, 1903),
    YunInfo(3, Yuan.SHANG, 1904, 1923),
    YunInfo(4, Yuan.ZHONG, 1924, 1943),
    YunInfo(5, Yuan.ZHONG, 1944, 1963),
    YunInfo(6, Yuan.ZHONG, 1964, 1983),
    YunInfo(7, Yuan.XIA,   1984, 2003),
    YunInfo(8, Yuan.XIA,   2004, 2023),
    YunInfo(9, Yuan.XIA,   2024, 2043),
]


def year_to_yun(year: int) -> YunInfo:
    """给定年份 → 所在运。"""
    for info in YUN_TABLE:
        if info.year_start <= year <= info.year_end:
            return info
    raise ValueError(
        f"\u5e74\u4efd {year} \u4e0d\u5728\u4e09\u5143\u4e5d\u8fd0 1864-2043 \u8303\u56f4\u5185"
    )


# 后天八卦中心方位（北=0°，东=90°，顺时针）
GUA_DEG: dict[str, float] = {
    "\u574e":   0.0,   # 坎
    "\u826e":  45.0,   # 艮
    "\u9707":  90.0,   # 震
    "\u5dfd": 135.0,   # 巽
    "\u79bb": 180.0,   # 离
    "\u5764": 225.0,   # 坤
    "\u5151": 270.0,   # 兑
    "\u4e7e": 315.0,   # 乾
}


def shan_to_gua(shan: str) -> str:
    """廿四山 → 后天八卦（按中心方位最近匹配）。"""
    from engine.core.compass_directions import SHAN_TABLE
    deg = SHAN_TABLE.get(shan, (0.0, ""))[0]
    best = "\u574e"
    best_diff = 360.0
    for g, c in GUA_DEG.items():
        d = abs(((deg - c + 180.0) % 360.0) - 180.0)
        if d < best_diff:
            best_diff = d
            best = g
    return best


def shan_facing_gua(facing_shan: str) -> str:
    """向 → 向所在的后天八卦（非对宫）。

    修复：旧实现错误地对 facing 取了对宫，导致向午(180°)被记成坎。
    """
    from engine.core.compass_directions import SHAN_TABLE
    f = SHAN_TABLE.get(facing_shan, (0.0, ""))[0]
    best = "\u574e"
    best_diff = 360.0
    for g, c in GUA_DEG.items():
        d = abs(((f - c + 180.0) % 360.0) - 180.0)
        if d < best_diff:
            best_diff = d
            best = g
    return best


# 八卦阴阳（用于「阳顺阴逆」判别）。
# 父卦（乾/兑/离/震）为阳；母卦（巽/坎/艮/坤）为阴。
GUA_YIN_YANG: dict[str, str] = {
    "\u574e": "\u9634",   # 坎
    "\u826e": "\u9634",   # 艮
    "\u9707": "\u9633",   # 震
    "\u5dfd": "\u9634",   # 巽
    "\u79bb": "\u9633",   # 离
    "\u5764": "\u9634",   # 坤
    "\u5151": "\u9633",   # 兑
    "\u4e7e": "\u9633",   # 乾   【修复】原误为阴
}


# 紫白顺飞顺序（洛书轨迹）：
# 中 → 乾 → 兑 → 艮 → 离 → 坎 → 坤 → 震 → 巽
# 对应洛书数序 5→6→7→8→9→1→2→3→4 所在宫
LUO_SHUN_ORDER = ["\u4e2d", "\u4e7e", "\u5151", "\u826e",
                  "\u79bb", "\u574e", "\u5764", "\u9707", "\u5dfd"]
# 阴逆 = 中 + 顺飞路径倒序
LUO_NI_ORDER = ["\u4e2d"] + list(reversed(LUO_SHUN_ORDER[1:]))

# 元旦盘：洛书本宫数 → 卦（5 居中无卦）
YUAN_DAN_STAR_GUA: dict[int, str] = {
    1: "\u574e",  # 坎
    2: "\u5764",  # 坤
    3: "\u9707",  # 震
    4: "\u5dfd",  # 巽
    5: "\u4e2d",  # 中
    6: "\u4e7e",  # 乾
    7: "\u5151",  # 兑
    8: "\u826e",  # 艮
    9: "\u79bb",  # 离
}

# 每卦三山（地元 / 天元 / 人元）— 无常派地-天-人序
GUA_SHAN_BY_YUAN: dict[str, dict[str, str]] = {
    "\u574e": {"地": "\u58ec", "天": "\u5b50", "人": "\u7678"},  # 坎：壬子癸
    "\u826e": {"地": "\u4e11", "天": "\u826e", "人": "\u5bc5"},  # 艮：丑艮寅
    "\u9707": {"地": "\u7532", "天": "\u536f", "人": "\u4e59"},  # 震：甲卯乙
    "\u5dfd": {"地": "\u8fb0", "天": "\u5dfd", "人": "\u5df3"},  # 巽：辰巽巳
    "\u79bb": {"地": "\u4e19", "天": "\u5348", "人": "\u4e01"},  # 离：丙午丁
    "\u5764": {"地": "\u672a", "天": "\u5764", "人": "\u7533"},  # 坤：未坤申
    "\u5151": {"地": "\u5e9a", "天": "\u9149", "人": "\u8f9b"},  # 兑：庚酉辛
    "\u4e7e": {"地": "\u620c", "天": "\u4e7e", "人": "\u4ea5"},  # 乾：戌乾亥
}

# 二十四山阴阳（玄空三元龙表；与 compass_directions 一致）
# 阳：乾坤艮巽 + 甲庚壬丙 + 寅申巳亥
# 阴：子午卯酉 + 辰戌丑未 + 乙辛丁癸
SHAN_YIN_YANG: dict[str, str] = {
    "\u58ec": "\u9633", "\u5b50": "\u9634", "\u7678": "\u9634",  # 壬阳 子阴 癸阴
    "\u4e11": "\u9634", "\u826e": "\u9633", "\u5bc5": "\u9633",  # 丑阴 艮阳 寅阳
    "\u7532": "\u9633", "\u536f": "\u9634", "\u4e59": "\u9634",  # 甲阳 卯阴 乙阴
    "\u8fb0": "\u9634", "\u5dfd": "\u9633", "\u5df3": "\u9633",  # 辰阴 巽阳 巳阳
    "\u4e19": "\u9633", "\u5348": "\u9634", "\u4e01": "\u9634",  # 丙阳 午阴 丁阴
    "\u672a": "\u9634", "\u5764": "\u9633", "\u7533": "\u9633",  # 未阴 坤阳 申阳
    "\u5e9a": "\u9633", "\u9149": "\u9634", "\u8f9b": "\u9634",  # 庚阳 酉阴 辛阴
    "\u620c": "\u9634", "\u4e7e": "\u9633", "\u4ea5": "\u9633",  # 戌阴 乾阳 亥阳
}


def _star_seq(center: int, n: int = 9) -> list[int]:
    """从 center 起连续 n 颗星（1..9 环绕）。"""
    c = int(center)
    if c < 1 or c > 9:
        raise ValueError(f"star center 须在 1..9，got {center}")
    return [((c - 1 + i) % 9) + 1 for i in range(n)]


def fly_stars(center_star: int, *, reverse: bool = False) -> dict[str, int]:
    """将 center_star 置中宫，沿洛书轨迹顺/逆飞满九宫。

    Returns:
        dict 宫名 → 星数（含「中」）
    """
    path = LUO_NI_ORDER if reverse else LUO_SHUN_ORDER
    stars = _star_seq(center_star, len(path))
    return {g: s for g, s in zip(path, stars)}


def period_plate(yun: int) -> dict[str, int]:
    """运盘（天盘）：当运星入中，**一律顺飞**（无常派下卦共识）。"""
    if yun < 1 or yun > 9:
        raise ValueError(f"yun 须在 1..9，got {yun}")
    return fly_stars(yun, reverse=False)


def shan_polarity(shan: str) -> str:
    """二十四山 → 玄空阴阳（阳/阴）。"""
    if shan in SHAN_YIN_YANG:
        return SHAN_YIN_YANG[shan]
    g = shan_to_gua(shan)
    return GUA_YIN_YANG.get(g, "\u9633")


def san_yuan_of(shan: str) -> str:
    """山 → 三元龙（天/地/人）。真源 compass_directions.SHAN_TABLE。"""
    from engine.core.compass_directions import SHAN_TABLE

    if shan in SHAN_TABLE:
        return SHAN_TABLE[shan][1]
    return "天"


def star_fly_polarity(center_star: int, reference_shan: str) -> str:
    """山盘/向盘顺逆所用阴阳（无常派下卦）。

    规则：
      1. 入中星 S=5（五黄）→ 取 reference_shan（坐/向）本身阴阳（D2）
      2. 否则 S → 元旦盘卦宫 → 该卦中与 reference **同元龙**之山 → 其阴阳
    """
    s = int(center_star)
    if s < 1 or s > 9:
        raise ValueError(f"center_star 须在 1..9，got {center_star}")
    if s == 5:
        return shan_polarity(reference_shan)

    yuan_gua = YUAN_DAN_STAR_GUA[s]
    yuan = san_yuan_of(reference_shan)
    matched = GUA_SHAN_BY_YUAN.get(yuan_gua, {}).get(yuan)
    if matched is None:
        return shan_polarity(reference_shan)
    return shan_polarity(matched)


def _detect_fan_fu_yin(m_chart: dict[str, int], f_chart: dict[str, int]) -> str:
    """粗检反吟/伏吟（山向两盘相对地盘关系的简化版）。

    伏吟：山盘与向盘九宫星数全同。
    反吟：山盘与向盘对应宫星数之和均为 10（洛书对宫）。
    完整反伏吟尚需对照运盘；此处仅作提示字段。
    """
    guas = [g for g in LUO_SHUN_ORDER if g != "\u4e2d"]
    if all(m_chart.get(g) == f_chart.get(g) for g in guas + ["\u4e2d"]):
        return "伏吟"
    if all(
        (m_chart.get(g, 0) + f_chart.get(g, 0)) == 10
        for g in guas + ["\u4e2d"]
    ):
        return "反吟"
    return ""


@dataclass
class XuanKongChart:
    """玄空九星盘。

    simplified=True：仅卦位/元运，星数不可信。
    simplified=False：基础三盘飞星已排；替卦/城门/零正神仍未完整。
    """

    yun: int
    yuan: str
    shan: str                # 坐山（用户输入）
    facing: str              # 向（用户输入）
    shan_gua: str            # 坐山所入后天卦
    facing_gua: str          # 向所在后天卦（非对宫）
    shan_star_at_facing: int | None  # 山星到向宫
    facing_star_at_facing: int | None  # 向星到向宫
    mountain_chart: dict[str, int] = field(default_factory=dict)
    facing_chart: dict[str, int] = field(default_factory=dict)
    period_chart: dict[str, int] = field(default_factory=dict)
    notes: str = ""
    simplified: bool = True  # True=仅卦位；False=基础飞星（仍非全规则）
    fan_fu_yin: str = ""     # "" / 伏吟 / 反吟（粗检）
    features_missing: list[str] = field(default_factory=list)


def fly_chart(yun: int, shan: str, facing: str) -> XuanKongChart:
    """生成沈氏玄空九星盘（简化）。

    警告⚠️：此实现为参考性简化盘。完整沈氏玄空排盘需分山、向两盘独立起中、
    阴阳顺逆、替卦、反伏吟、城门、零神正神等繁复规则，与真实堪舆实务差距显著。

    仅输出运 + 元 + 山的 24 山 × 9 运（216局）的简化卦位，不输出星数，
    以避免给出错误的"山星/向星"误导用户。

    如需基础三盘飞星，请用 fly_chart_strict（仍非全规则，xuankong_implemented 仍为 false）。
    """
    if yun < 1 or yun > 9:
        raise ValueError(f"yun 须在 1..9，got {yun}")
    info = year_to_yun(1864 + (yun - 1) * 20)
    shan_g = shan_to_gua(shan)
    fac_g = shan_facing_gua(facing)

    # 仅做"卦位"映射；不输出数字星数（避免错误）
    m_chart = {g: 0 for g in GUA_DEG}
    f_chart = {g: 0 for g in GUA_DEG}

    return XuanKongChart(
        yun=yun,
        yuan=info.yuan.value,
        shan=shan,
        facing=facing,
        shan_gua=shan_g,
        facing_gua=fac_g,
        shan_star_at_facing=None,   # 【明确】简化盘不输出星数
        facing_star_at_facing=None,
        mountain_chart=m_chart,
        facing_chart=f_chart,
        period_chart={},
        notes=(
            "玄空简化盘：只输出卦位 / 元 / 运；星数请用 fly_chart_strict。"
            "替卦 / 城门 / 零正神仍未实现。"
        ),
        simplified=True,
        features_missing=["stars", "替卦", "反伏吟完整", "城门", "零神正神"],
    )


# 沈氏/无常派常见「兼向替星」表（研究用；流派差异大，非唯一真源）
# 含义：山盘或向盘入中原星 → 替入星（当坐/向判定为兼向时启用）
TI_STAR_TABLE: dict[int, int] = {
    1: 6,  # 一白兼 → 六白替
    2: 7,  # 二黑兼 → 七赤替
    3: 8,  # 三碧兼 → 八白替
    4: 9,  # 四绿兼 → 九紫替
    5: 2,  # 五黄兼 → 二黑替（一说 8；取阳宅常用 2）
    6: 1,  # 六白兼 → 一白替
    7: 2,  # 七赤兼 → 二黑替
    8: 3,  # 八白兼 → 三碧替
    9: 4,  # 九紫兼 → 四绿替
}


def detect_jian_xiang(
    shan: str,
    *,
    deg_override: float | None = None,
    jian_threshold_deg: float = 3.0,
) -> tuple[bool, float, str]:
    """判断二十四山是否「兼向」。

    规则：实际方位与本山中心角差 ≥ jian_threshold_deg（默认 3°）则兼。
    若无 deg_override，仅按山名无法知兼 → 返回 (False, 0, "正针")。

    Returns:
        (is_jian, offset_deg, note)
    """
    from engine.core.compass_directions import SHAN_TABLE

    if shan not in SHAN_TABLE:
        return False, 0.0, "未知山"
    center = float(SHAN_TABLE[shan][0])
    if deg_override is None:
        return False, 0.0, "无实测角，按下卦"
    d = abs(((float(deg_override) - center + 180.0) % 360.0) - 180.0)
    if d >= jian_threshold_deg:
        return True, float(d), f"兼{d:.1f}°"
    return False, float(d), "正针"


def substitute_ti_star(center_star: int) -> int:
    """兼向时入中星替入。"""
    s = int(center_star)
    if s < 1 or s > 9:
        raise ValueError(f"center_star 须在 1..9，got {center_star}")
    return int(TI_STAR_TABLE.get(s, s))


def fly_chart_strict(
    yun: int,
    shan: str,
    facing: str,
    *,
    shan_deg: float | None = None,
    facing_deg: float | None = None,
    force_ti: bool | None = None,
) -> XuanKongChart:
    """无常派下卦/替卦三盘飞星（运盘 + 山盘 + 向盘）。

    已实现（B8 + P2 替卦）：
      - 运星入中，**一律顺飞**
      - 坐/向宫运星入中；顺逆 = 星→元旦盘同元龙阴阳
      - 五黄入中取坐/向本阴阳（D2）
      - 山星/向星到向宫、粗反伏吟提示
      - **兼向替卦**：坐或向兼 ≥3° 时，对应盘入中星走替星表再飞

    未实现（故对外仍禁止 xuankong_implemented=true）：
      - 完整反伏吟对照运盘
      - 城门诀、零神/正神、三般卦等
      - 替卦全流派细则（仅沈氏常见替星表）

    研究/对照用；勿当作专业排盘软件输出。
    """
    if yun < 1 or yun > 9:
        raise ValueError(f"yun 须在 1..9，got {yun}")
    info = year_to_yun(1864 + (yun - 1) * 20)
    shan_g = shan_to_gua(shan)
    fac_g = shan_facing_gua(facing)

    jian_shan, off_s, note_s = detect_jian_xiang(shan, deg_override=shan_deg)
    jian_fac, off_f, note_f = detect_jian_xiang(facing, deg_override=facing_deg)
    use_ti = bool(force_ti) if force_ti is not None else (jian_shan or jian_fac)

    # 1. 运盘：一律顺飞（替卦不改运盘）
    p_chart = period_plate(yun)

    # 2. 山星：运盘坐宫星入中；兼向则替星
    m_center = int(p_chart.get(shan_g, yun))
    m_ti_applied = False
    if use_ti and (jian_shan or force_ti):
        m_center = substitute_ti_star(m_center)
        m_ti_applied = True
    m_pol = star_fly_polarity(m_center, shan)
    m_reverse = m_pol == "\u9634"
    m_chart = fly_stars(m_center, reverse=m_reverse)

    # 3. 向星：运盘向宫星入中；兼向则替星
    f_center = int(p_chart.get(fac_g, yun))
    f_ti_applied = False
    if use_ti and (jian_fac or force_ti):
        f_center = substitute_ti_star(f_center)
        f_ti_applied = True
    f_pol = star_fly_polarity(f_center, facing)
    f_reverse = f_pol == "\u9634"
    f_chart = fly_stars(f_center, reverse=f_reverse)

    shan_star_at_facing = int(m_chart.get(fac_g, 0))
    facing_star_at_facing = int(f_chart.get(fac_g, 0))
    fan_fu = _detect_fan_fu_yin(m_chart, f_chart)

    missing = ["反伏吟完整(对照运盘)", "城门", "零神正神", "三般卦"]
    if not use_ti:
        # 正针下卦；替卦能力已具备但本盘未触发
        pass
    else:
        # 替卦已应用；全流派细则仍缺
        missing.append("替卦全流派细则")

    notes_parts = [
        f"{'替卦' if use_ti else '下卦'}三盘：运={yun}入中(顺飞)",
        f"山星起{m_center}({'阴逆' if m_reverse else '阳顺'}"
        f"{'·替' if m_ti_applied else ''}，同元龙；{note_s})",
        f"向星起{f_center}({'阴逆' if f_reverse else '阳顺'}"
        f"{'·替' if f_ti_applied else ''}，同元龙；{note_f})",
    ]
    if fan_fu:
        notes_parts.append(f"粗检{fan_fu}")
    notes_parts.append("城门/零正神未实现；研究用，勿宣称专业排盘。")

    return XuanKongChart(
        yun=yun,
        yuan=info.yuan.value,
        shan=shan,
        facing=facing,
        shan_gua=shan_g,
        facing_gua=fac_g,
        shan_star_at_facing=shan_star_at_facing,
        facing_star_at_facing=facing_star_at_facing,
        mountain_chart={k: v for k, v in m_chart.items() if k != "\u4e2d"},
        facing_chart={k: v for k, v in f_chart.items() if k != "\u4e2d"},
        period_chart={k: v for k, v in p_chart.items() if k != "\u4e2d"},
        notes="；".join(notes_parts),
        simplified=False,
        fan_fu_yin=fan_fu,
        features_missing=missing,
    )


def wang_xiang(yun: int, chart: XuanKongChart) -> dict[str, bool]:
    """判别"旺向/旺山"：山星或向星到向宫 = 令星（yun）。

    双星会向 → 旺；五黄到向宫（star == 5）则需换替。
    简化盘（星数为 None/0）一律返回 False。
    """
    ss = chart.shan_star_at_facing
    fs = chart.facing_star_at_facing
    if ss is None or fs is None or (ss == 0 and fs == 0 and chart.simplified):
        return {
            "wang_xiang": False,
            "wang_shan": False,
            "five_yellow_at_facing": False,
        }
    wang = bool(ss == yun or fs == yun)
    return {
        "wang_xiang": wang,
        "wang_shan": bool(chart.mountain_chart.get(chart.shan_gua, 0) == yun),
        "five_yellow_at_facing": bool(ss == 5 or fs == 5),
    }


def score_yun(yun: int, chart: XuanKongChart) -> int:
    """基于九运与玄空盘给出 0-100 评分。

    简化盘无有效星数时返回中性 50，避免虚假高低分。
    """
    if chart.simplified or chart.shan_star_at_facing is None:
        return 50
    base = 60
    flags = wang_xiang(yun, chart)
    if flags["wang_xiang"]:
        base += 18
    if flags["wang_shan"]:
        base += 8
    if flags["five_yellow_at_facing"]:
        base -= 25
    # 双星会向加分
    if (
        chart.shan_star_at_facing == chart.facing_star_at_facing
        and chart.shan_star_at_facing == yun
    ):
        base += 8
    return int(max(0, min(100, base)))
