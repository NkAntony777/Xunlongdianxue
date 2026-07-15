"""Scoring weight tables."""
from __future__ import annotations

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


